from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from openai import OpenAI
import json, os, tempfile, uuid, librosa

app = FastAPI()

# 创建图片存储目录
os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

# 初始化模型
whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
client = OpenAI(
    api_key="sk-d958fd9b10c6455485ab417e4a5699ef",  # 你的 API Key
    base_url="https://api.deepseek.com"  # 或国内兼容地址，如通义千问：https://dashscope.aliyuncs.com/compatible-mode/v1
)

# ============ 语音转文字 (鲁棒版) ============
def speech_to_text(audio_bytes: bytes) -> str:
    # 将音频字节写入临时文件，保留原始后缀，让librosa自动处理
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 用 librosa 加载，自动重采样到 16kHz 单声道
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
        os.unlink(tmp_path) # 尽早删除临时文件

        # 直接传 numpy 数组给 faster-whisper
        segments, _ = whisper.transcribe(audio, language="zh")
        return "".join([seg.text for seg in segments])
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return "" # 返回空字符串，避免服务崩溃