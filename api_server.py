from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from openai import OpenAI
import json, os, tempfile, uuid

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

# ============ 语音转文字 ============
def speech_to_text(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    segments, _ = whisper.transcribe(tmp_path, language="zh")
    os.unlink(tmp_path)
    return "".join([seg.text for seg in segments])

# ============ LLM生成待办 ============
def generate_todos(text: str) -> list:
    system_prompt = """你是一个智能待办助手。将用户的语音文字解析为 JSON 数组，字段：title, deadline, category, priority, notes。只输出 JSON 数组。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"system","content":system_prompt},{"role":"user","content":text}],
        temperature=0.1
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```json"): content = content[7:-3]
    return json.loads(content)

# ============ API：语音 → 待办 ============
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

    # 3. LLM生成待办
    todos = generate_todos(audio_text)
    if image_url:
        todos[0]["image_url"] = image_url

    return JSONResponse({"text": audio_text, "todos": todos})

# ============ 前端H5页面 ============
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no">
    <title>AI 记忆外挂</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:-apple-system, sans-serif; background:#f5f5f5; padding:20px; max-width:500px; margin:auto; }
        h1 { text-align:center; color:#333; margin-bottom:20px; }
        .card { background:white; border-radius:16px; padding:20px; margin-bottom:16px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }
        button { width:100%; padding:14px; border:none; border-radius:12px; font-size:18px; font-weight:bold; cursor:pointer; margin:8px 0; }
        .btn-record { background:#ff4757; color:white; }
        .btn-record.recording { background:#ff6b81; animation:pulse 1.5s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }
        .btn-submit { background:#2ed573; color:white; }
        .btn-upload { background:#1e90ff; color:white; }
        #result { margin-top:16px; }
        .todo-item { background:white; border-radius:12px; padding:16px; margin-bottom:10px; display:flex; align-items:center; box-shadow:0 1px 4px rgba(0,0,0,0.08); }
        .todo-item img { width:60px; height:60px; border-radius:8px; object-fit:cover; margin-right:12px; }
        .todo-info { flex:1; }
        .todo-info h3 { font-size:16px; margin-bottom:4px; }
        .todo-info p { font-size:13px; color:#666; }
        input[type="text"] { width:100%; padding:14px; border:1px solid #ddd; border-radius:12px; font-size:16px; }
        #preview { text-align:center; margin:10px 0; }
        #preview img { max-width:200px; border-radius:12px; }
    </style>
</head>
<body>
    <h1>🧠 AI 记忆外挂</h1>

    <!-- 录音 -->
    <div class="card">
        <h3>🎤 语音输入</h3>
        <button class="btn-record" id="recordBtn">按住录音</button>
        <p id="status" style="text-align:center;margin-top:10px;color:#666;"></p>
    </div>

    <!-- 文字输入 -->
    <div class="card">
        <h3>✏️ 手动输入</h3>
        <input type="text" id="textInput" placeholder="输入你想记的事..." />
    </div>

    <!-- 图片上传 -->
    <div class="card">
        <h3>📷 照片（可选）</h3>
        <input type="file" id="imageInput" accept="image/*" capture="environment" style="display:none" />
        <button class="btn-upload" onclick="document.getElementById('imageInput').click()">选择照片</button>
        <div id="preview"></div>
    </div>

    <!-- 生成按钮 -->
    <button class="btn-submit" onclick="submitTodo()">✨ 生成待办</button>

    <!-- 结果 -->
    <div id="result"></div>

    <script>
        let mediaRecorder, audioChunks = [];
        const recordBtn = document.getElementById("recordBtn");

        recordBtn.addEventListener("touchstart", async (e) => {
            e.preventDefault();
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                stream.getTracks().forEach(t => t.stop());
                document.getElementById("status").textContent = "录音完成，点击「生成待办」提交";
            };
            audioChunks = [];
            mediaRecorder.start();
            recordBtn.classList.add("recording");
            recordBtn.textContent = "录音中，松开发送";
            document.getElementById("status").textContent = "正在录音...";
        });

        recordBtn.addEventListener("touchend", () => {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                mediaRecorder.stop();
                recordBtn.classList.remove("recording");
                recordBtn.textContent = "按住录音";
            }
        });

        document.getElementById("imageInput").addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = ev => document.getElementById("preview").innerHTML = `<img src="${ev.target.result}" />`;
                reader.readAsDataURL(file);
            }
        });

        async function submitTodo() {
            const formData = new FormData();
            const text = document.getElementById("textInput").value;
            const image = document.getElementById("imageInput").files[0];

            if (audioChunks.length > 0) {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                formData.append("audio", audioBlob, "recording.wav");
            } else if (text) {
                formData.append("text", text);
            } else {
                alert("请先录音或输入文字");
                return;
            }

            if (image) formData.append("image", image);

            document.getElementById("result").innerHTML = "<p style='text-align:center'>⏳ 正在处理...</p>";

            const resp = await fetch("/api/todo", { method: "POST", body: formData });
            const data = await resp.json();

            let html = `<p style="text-align:center;color:#666;">🎤 识别：${data.text}</p>`;
            data.todos.forEach((t, i) => {
                html += `<div class="todo-item">`;
                if (t.image_url) html += `<img src="${t.image_url}" />`;
                html += `<div class="todo-info"><h3>${t.title}</h3><p>⏰ ${t.deadline || "无"} | 📁 ${t.category || "未知"} | ${{"高":"🔴","中":"🟡","低":"🟢"}[t.priority] || ""}${t.priority}</p></div></div>`;
            });
            document.getElementById("result").innerHTML = html;
            audioChunks = [];
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)