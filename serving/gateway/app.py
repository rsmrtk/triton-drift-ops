"""
Inference gateway: the piece that connects Triton to the drift monitor.

Triton serves raw logits and knows nothing about drift. This gateway sits
in front of it: accepts an image, preprocesses, calls Triton, computes
softmax confidence — and feeds every confidence score into the
DriftMonitor, which exposes the drift metrics Prometheus scrapes and the
alert rules fire on. Without this component the drift loop isn't wired to
live traffic at all.

The baseline confidence distribution is produced at training time
(training/train.py --log-mlflow logs it as an artifact) and mounted/
downloaded here at startup. If it's missing, the gateway still serves
inference but reports no drift — same graceful-degradation stance as the
rest of the stack.

Usage:
    TRITON_URL=localhost:8000 uvicorn app:app --port 8081
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from prometheus_client import make_asgi_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TRITON_URL = os.getenv("TRITON_URL", "triton:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "traffic_sign_classifier")
BASELINE_PATH = os.getenv("BASELINE_PATH", "/baseline/confidences.npy")
INPUT_SIZE = 48

NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

monitor = None
triton = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global monitor, triton
    triton = httpclient.InferenceServerClient(url=TRITON_URL)

    baseline_file = Path(BASELINE_PATH)
    if baseline_file.exists():
        from drift.monitor import DriftMonitor

        baseline = np.load(baseline_file)
        monitor = DriftMonitor(baseline_confidences=baseline)
        logger.info("drift monitor armed with %d baseline confidences", len(baseline))
    else:
        logger.warning(
            "no baseline at %s — serving without drift detection", BASELINE_PATH
        )
    yield


app = FastAPI(title="inference-gateway", lifespan=lifespan)
# /metrics for Prometheus — includes the drift gauges DriftMonitor sets
app.mount("/metrics", make_asgi_app())


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - NORMALIZE_MEAN) / NORMALIZE_STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    input_data = preprocess(await file.read())

    infer_input = httpclient.InferInput("input", input_data.shape, "FP32")
    infer_input.set_data_from_numpy(input_data)

    try:
        result = triton.infer(model_name=MODEL_NAME, inputs=[infer_input])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"triton: {e}")

    logits = result.as_numpy("logits")[0]
    probs = softmax(logits)
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    if monitor is not None:
        monitor.observe(np.array([confidence]))

    return {"class_id": pred_class, "confidence": round(confidence, 4)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "drift_monitor_armed": monitor is not None,
    }
