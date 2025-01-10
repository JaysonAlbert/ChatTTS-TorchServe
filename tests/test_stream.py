import ChatTTS
import logging
import io
import torchaudio
import torch

chat = ChatTTS.Chat(logging.getLogger("ChatTTS"))
chat.load(source="huggingface")

pa = {}

params = {
    "text": ["你好", "代销上午流程deadline了，今天托管行交收金额11,234.23元，完成比例99%"],
    "stream": True,
    "lang": "zh",
    "use_decoder": False,
    "do_text_normalization": True,
    "do_homophone_replacement": False,
    "skip_refine_text": True,
    "params_refine_text": {"prompt": "[oral_0][break_6]"},
    "params_infer_code": {
        "prompt": "[speed_4]",
        "manual_seed": 928,
        "top_P": 0.2,
        "top_K": 15,
        "temperature": 0.15,
    },
}

def process():

    for i in range(1):
        pp = {
            "text": params.get("text"),
            "stream": params.get("stream", False),
            "lang": params.get("lang"),
            "skip_refine_text": params.get("skip_refine_text", False),
            "use_decoder": params.get("use_decoder", True),
            "do_text_normalization": params.get("do_text_normalization", True),
            "do_homophone_replacement": params.get("do_homophone_replacement", False),
            "params_refine_text": ChatTTS.Chat.RefineTextParams(),
            "params_infer_code": ChatTTS.Chat.InferCodeParams(),
        }

        yield chat.infer(**pp)

for wavs in process():
    print(type(wavs))
    for chunk in wavs:
        for wav in chunk:
            print(wav.shape)
            buf = io.BytesIO()
            torchaudio.save(
                                buf, torch.from_numpy(wav).unsqueeze(0), 24000, format="wav"
                            )
            buf.seek(0)
