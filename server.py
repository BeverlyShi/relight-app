import os
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Mock 模式 ──────────────────────────────────────
# 本地开发时设置环境变量 MOCK=1，不加载模型，返回假图片
MOCK = os.environ.get("MOCK", "0") == "1"

if not MOCK:
    from inference import run_relight
else:
    print("⚠️  MOCK 模式：不加载模型，返回原图")
    def run_relight(image, **kwargs):
        # mock 模式直接返回原图，方便本地调试接口逻辑
        return image

# ── FastAPI ───────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "mock": MOCK}

@app.post("/relight")
async def relight_api(
    file: UploadFile,
    angle: float = Form(0.0),
    brightness: float = Form(50.0),
    temperature: float = Form(5000.0),
    prompt: str = Form("natural lighting"),
    negative_prompt: str = Form("lowres, bad anatomy, bad hands, cropped, worst quality"),
    steps: int = Form(25),
    cfg: float = Form(2.0),
    seed: int = Form(12345),
    highres_denoise: float = Form(0.3),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = run_relight(
        image=image,
        angle_deg=angle,
        brightness=brightness,
        temperature=temperature,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg=cfg,
        seed=seed,
        highres_denoise=highres_denoise,
    )

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    host = os.environ.get("RELIGHT_HOST", "0.0.0.0")
    port = int(os.environ.get("RELIGHT_PORT", os.environ.get("PORT", "6007")))
    uvicorn.run(app, host=host, port=port)
