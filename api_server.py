from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from openai import OpenAI
import json, os, tempfile, uuid, librosa
import soundfile as sf

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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # 直接读取 WAV，前端已经确保是 16kHz 单声道
        audio, sr = sf.read(tmp_path)
        os.unlink(tmp_path)

        segments, _ = whisper.transcribe(audio, language="zh")
        text = "".join([seg.text for seg in segments])
        print(f"🎤 识别结果: {text}")
        return text
    except Exception as e:
        print(f"❌ 语音识别出错: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return ""

# ========== LLM 解析待办 ==========
def generate_todos(text: str) -> list:
    system_prompt = """
你是一个智能待办助手。将用户的语音文字解析为严格的 JSON 数组。
字段：title（标题）、deadline（时间）、category（工作/生活/学习）、priority（高/中/低）、notes（补充说明）。
只输出 JSON 数组，不要其他文字。"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.1
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
    return json.loads(content)


# ========== API：生成待办 ==========
@app.post("/api/todo")
async def create_todo(
        audio: UploadFile = File(None),
        image: UploadFile = File(None),
        text: str = Form(None)
):
    # 1. 获取文字
    if audio:
        audio_text = speech_to_text(await audio.read())
    else:
        audio_text = text or ""

    # 2. 处理图片
    image_url = None
    if image:
        img_data = await image.read()
        ext = image.filename.split(".")[-1] if image.filename else "jpg"
        img_name = f"{uuid.uuid4().hex}.{ext}"
        img_path = os.path.join("images", img_name)
        with open(img_path, "wb") as f:
            f.write(img_data)
        image_url = f"/images/{img_name}"

    # 3. LLM 生成待办
    todos = generate_todos(audio_text)
    if image_url and len(todos) > 0:
        todos[0]["image_url"] = image_url

    return JSONResponse({"text": audio_text, "todos": todos})


# ========== 最后挂载前端静态文件（必须放在 API 路由之后）==========
app.mount("/", StaticFiles(directory="static", html=True), name="static")