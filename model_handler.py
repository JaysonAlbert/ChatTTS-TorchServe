import os
import json
import types
import torch
from typing import Callable
from functools import partial
import io
from ts.torch_handler.base_handler import BaseHandler
import logging
import torchaudio
from ts.handler_utils.utils import send_intermediate_predict_response

import ChatTTS

logger = logging.getLogger("TorchServeHandler")

from typing import Callable


def normalizer_zh_tn() -> Callable[[str], str]:
    from tn.chinese.normalizer import Normalizer

    return Normalizer(remove_interjections=False).normalize


def normalizer_en_nemo_text() -> Callable[[str], str]:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    return partial(
        Normalizer(input_case="cased", lang="en").normalize,
        verbose=False,
        punct_post_process=True,
    )


class ChatTTSHandler(BaseHandler):
    def __init__(self):
        super(ChatTTSHandler, self).__init__()
        self.chat = None
        self.initialized = False

    def initialize(self, ctx):
        """Load the model and initialize resources."""
        logger.info("Initializing ChatTTS...")
        self.chat = ChatTTS.Chat(logging.getLogger("ChatTTS"))
        self.chat.normalizer.register("en", normalizer_en_nemo_text())
        self.chat.normalizer.register("zh", normalizer_zh_tn())

        model_dir = ctx.system_properties.get("model_dir")
        os.chdir(model_dir)
        if self.chat.load(source="custom", custom_path=model_dir, compile=True):
            logger.info("Models loaded successfully.")
        else:
            logger.error("Models load failed.")
            raise RuntimeError("Failed to load models.")
        self.initialized = True

    def preprocess(self, data):
        """Preprocess incoming requests."""
        if len(data) == 0:
            raise ValueError("No data received for inference.")
        logger.info(f"batch size: {len(data)}")
        return self._group_reuqest_by_config(data)

    def _group_reuqest_by_config(self, data):

        batched_requests = {}
        for req in data:
            params = req.get("body")
            text = params.pop("text")

            key = json.dumps(params)

            if key not in batched_requests:
                params_refine_text = params.get("params_refine_text")
                params_infer_code = params.get("params_infer_code")

                if (
                    params_infer_code
                    and params_infer_code.get("manual_seed") is not None
                ):
                    torch.manual_seed(params_infer_code.get("manual_seed"))
                    params_infer_code["spk_emb"] = self.chat.sample_random_speaker()

                batched_requests[key] = {
                    "text": [text],
                    "stream": params.get("stream", False),
                    "lang": params.get("lang"),
                    "skip_refine_text": params.get("skip_refine_text", False),
                    "use_decoder": params.get("use_decoder", True),
                    "do_text_normalization": params.get("do_text_normalization", True),
                    "do_homophone_replacement": params.get(
                        "do_homophone_replacement", False
                    ),
                    "params_refine_text": (
                        ChatTTS.Chat.RefineTextParams(**params_refine_text)
                        if params_refine_text
                        else ChatTTS.Chat.RefineTextParams()
                    ),
                    "params_infer_code": (
                        ChatTTS.Chat.InferCodeParams(**params_infer_code)
                        if params_infer_code
                        else ChatTTS.Chat.InferCodeParams()
                    ),
                }
            else:
                batched_requests[key]["text"].append(text)

        return batched_requests

    def inference(self, data):
        """Run inference."""

        for key, params in data.items():
            logger.debug(f"Request: {key}")
            logger.debug(f"Text input: {str(params['text'])}")

            yield self.chat.infer(**params)

    def postprocess(self, batched_results):
        """Post-process inference results into raw wav data."""
        results = []
        for wavs in batched_results:
            if isinstance(wavs, types.GeneratorType):  # stream mode
                for chunk in wavs:
                    bufs = []
                    for wav in chunk:
                        buf = io.BytesIO()
                        torchaudio.save(
                            buf,
                            torch.from_numpy(wav).unsqueeze(0),
                            24000,
                            format="wav",
                            bits_per_sample=16,
                        )
                        buf.seek(0)
                        bufs.append(buf.getvalue())
                    logger.info(
                        f"Sending intermediate result, request_ids: {self.context.request_ids}"
                    )
                    send_intermediate_predict_response(
                        bufs,
                        self.context.request_ids,
                        "audio/wav",
                        200,
                        self.context,
                    )
                # return empty result to avoid returning the same result multiple times
                for i in range(len(self.context.request_ids)):
                    results.append(b"")
            else:
                for wav in wavs:
                    buf = io.BytesIO()
                    torchaudio.save(
                        buf, torch.from_numpy(wav).unsqueeze(0), 24000, format="wav"
                    )
                    buf.seek(0)
                    results.append(buf.getvalue())

        return results
