from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "index.html"
MODEL_PATH = ROOT / "yolo11n.onnx"
INPUT_SIZE = 640
MAX_FRAME_BYTES = 2 * 1024 * 1024

app = FastAPI(title="SniffStation Scout")
session = None
input_name = None
model_error = "ONNX model has not loaded."


@app.on_event("startup")
def load_model():
    global session, input_name, model_error
    try:
        session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        model_error = ""
    except Exception as exc:
        model_error = f"Could not load ONNX model: {exc}"


@app.get("/")
def home():
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_PATH)


@app.get("/api/status")
def status():
    return {
        "ready": session is not None,
        "message": model_error or "ONNX dog detector is ready.",
    }


def prepare_image(image):
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)

    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)

    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized

    # Camera frames from OpenCV are BGR; YOLO expects RGB.
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = tensor.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    return tensor, scale, pad_x, pad_y, width, height


def find_dog(output):
    prediction = np.squeeze(output)

    if prediction.ndim != 2:
        raise ValueError(f"Unexpected ONNX output shape: {output.shape}")

    # Raw YOLO11 detection output is generally [84, anchors].
    # Convert it to [anchors, 84].
    if prediction.shape[0] == 84 and prediction.shape[1] > 84:
        prediction = prediction.T

    # If export includes NMS, rows are x1, y1, x2, y2, score, class.
    if prediction.shape[1] == 6:
        dogs = prediction[prediction[:, 5].astype(int) == 16]
        if len(dogs) == 0:
            return None
        row = dogs[np.argmax(dogs[:, 4])]
        confidence = float(row[4])
        if confidence < 0.30:
            return None
        x1, y1, x2, y2 = row[:4]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y, confidence

    # Raw YOLO11 output: four box values followed by 80 class scores.
    if prediction.shape[1] < 21:
        raise ValueError(f"Unexpected YOLO output shape: {output.shape}")

    dog_scores = prediction[:, 4 + 16]
    best = int(np.argmax(dog_scores))
    confidence = float(dog_scores[best])
    if confidence < 0.30:
        return None

    center_x, center_y, box_width, box_height = prediction[best, :4]
    return float(center_x), float(center_y), confidence


@app.post("/api/analyze")
async def analyze(frame: UploadFile = File(...)):
    if session is None:
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
        tensor, scale, pad_x, pad_y, width, height = prepare_image(image)
        outputs = session.run(None, {input_name: tensor})
        detection = find_dog(outputs[0])

        if detection is None:
            return {"detected": False, "center": None, "confidence": 0.0}

        model_x, model_y, confidence = detection
        x = (model_x - pad_x) / scale / width
        y = (model_y - pad_y) / scale / height

        return {
            "detected": True,
            "center": [
                float(np.clip(x, 0.0, 1.0)),
                float(np.clip(y, 0.0, 1.0)),
            ],
            "confidence": round(confidence, 3),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
