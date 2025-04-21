import sounddevice as sd
import numpy as np
from funasr import AutoModel

# 初始化 FunASR 模型（ASR + VAD + PUNC）
asr_model = AutoModel(
    model="paraformer-zh-streaming",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    use_vad=True,
    use_punc=True,
    mode="online",
    use_gpu=False
)

# 音频参数
samplerate = 16000
block_size = 6400  # 相当于 0.4 秒的音频（建议配合 VAD）

print("🎙️ 开始说话吧，说完停一下就会显示结果（Ctrl+C 退出）...")

buffer = []

def callback(indata, frames, time, status):
    global buffer
    audio_chunk = indata[:, 0]
    buffer.extend(audio_chunk)

    # 每 0.4s 识别一次
    if len(buffer) >= block_size:
        audio_data = np.array(buffer[:block_size], dtype=np.float32)
        buffer = buffer[block_size:]

        res = asr_model.generate(input={"speech": audio_data, "is_final": False})
        if res:
            print("📝 实时结果:", res["text"])

try:
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback, blocksize=block_size):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("🛑 结束识别")