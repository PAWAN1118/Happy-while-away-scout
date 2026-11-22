from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "index.html"
MODEL_PATH = ROOT / "yolo11n.pt"
MAX_FRAME_BYTES = 2 * 1024 * 1024

app = FastAPI(title="SniffStation Scout")
model = None
model_error = "The dog detector has not loaded."


@app.on_event("startup")
def load_model():
    global model, model_error
    try:
        # The small pretrained weights are bundled with this deployment.
        model = YOLO(str(MODEL_PATH))
        model_error = ""
    except Exception as exc:
        model_error = f"Could not load the dog detector: {exc}"


@app.get("/")
def home():
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_PATH)


@app.get("/api/status")
def status():
    return {
        "ready": model is not None,
        "message": model_error or "Pretrained dog detector is ready.",
    }


@app.post("/api/analyze")
async def analyze(frame: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail=model_error)

    raw = await frame.read(MAX_FRAME_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image frame")
    if len(raw) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Image frame is too large")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image frame")

    try:
        result = model.predict(
            source=image,
            imgsz=640,
            conf=0.30,
            classes=[16],  # COCO class 16 is dog.
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return {"detected": False, "center": None, "confidence": 0.0}

        scores = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        best = int(np.argmax(scores))
        x1, y1, x2, y2 = boxes[best]
        height, width = image.shape[:2]

        return {
            "detected": True,
            "center": [
                float(((x1 + x2) / 2) / width),
                float(((y1 + y2) / 2) / height),
            ],
            "confidence": round(float(scores[best]), 3),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
