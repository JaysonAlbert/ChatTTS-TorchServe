import requests
import pyaudio
from pydub import AudioSegment
from io import BytesIO

# Prepare the data as a dictionary
data = {
    "text": "代销上午流程deadline了，今天托管行交收金额11,234.23元，完成比例99%",
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

# Prepare headers for the request
headers = {"Content-Type": "application/json"}

# URL to which the request will be made
url = "http://localhost:8080/predictions/chattts"

# Send POST request with JSON data
response = requests.post(url, json=data, headers=headers, stream=True)

# 打开音频流（配置为单声道、16位、24000采样率）

# 初始化pyaudio来播放音频
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=24000,
    output=True,
)

# 逐步读取并播放返回的音频流
chunk_size = 24000 * 2  # 1 second of audio data (24000 samples per second * 2 bytes per sample)
for chunk in response.iter_content(chunk_size=chunk_size):
    if chunk:
        audio = AudioSegment.from_wav(BytesIO(chunk))
        # 将音频块写入PyAudio流进行播放
        stream.write(audio.raw_data)

stream.stop_stream()
stream.close()
p.terminate()
