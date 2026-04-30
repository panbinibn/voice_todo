from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from openai import OpenAI
import json, os, tempfile, uuid, librosa
import soundfile as sf
import sqlite3
app = FastAPI()

# 创建图片存储目录
os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

# ----------------------- 数据库初始化 -----------------------
DB_PATH = os.path.join("/data", "todos.db")  # 使用 Railway 持久化 Volume 路径

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                deadline TEXT,
                category TEXT,
                priority TEXT,
                notes TEXT,
                image_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    print("数据库初始化完毕")

init_db()

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

# ----------------------- 数据库操作 -----------------------
def save_todos_to_db(user_id: str, todos: list, image_url: str = None):
    with sqlite3.connect(DB_PATH) as conn:
        for i, todo in enumerate(todos):
            # 只有第一条待办关联图片
            img = image_url if i == 0 and image_url else None
            conn.execute("""
                INSERT INTO todos (user_id, title, deadline, category, priority, notes, image_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                todo.get("title", ""),
                todo.get("deadline", "无"),
                todo.get("category", "未知"),
                todo.get("priority", "中"),
                todo.get("notes", ""),
                img
            ))
def get_user_todos(user_id: str) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM todos WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        # 返回字段名与前端匹配
        return [{
            "title": row["title"],
            "deadline": row["deadline"],
            "category": row["category"],
            "priority": row["priority"],
            "notes": row["notes"],
            "image_url": row["image_url"],
            "created_at": row["created_at"]
        } for row in rows]
# ----------------------- API 路由 -----------------------
@app.post("/api/todo")
async def create_todo(
        audio: UploadFile = File(None),
        image: UploadFile = File(None),
        text: str = Form(None),
        user_id: str = Form("")
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

    # 4. 如果有用户标识，保存到数据库
    if user_id and todos:
        save_todos_to_db(user_id, todos, image_url)

    return JSONResponse({"text": audio_text, "todos": todos})

@app.get("/api/todos")
async def get_todos(user_id: str = ""):
    if not user_id:
        return JSONResponse({"todos": []})
    return JSONResponse({"todos": get_user_todos(user_id)})

# ========== 最后挂载前端静态文件（必须放在 API 路由之后）==========
app.mount("/", StaticFiles(directory="static", html=True), name="static")