"""
Minimal Triton HTTP client — sends a preprocessed image to the deployed
traffic_sign_classifier model and prints the predicted class + confidence.

This is also the shape of code the drift monitor will wrap: every real
inference call goes through here, softmax confidence gets extracted and
fed to `drift.monitor.DriftMonitor.observe(...)`.

Usage:
    python client.py --image path/to/sign.png --url localhost:8000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from model.net import INPUT_SIZE  # noqa: E402

NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - NORMALIZE_MEAN) / NORMALIZE_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)  # add batch dim


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--url", type=str, default="localhost:8000")
    parser.add_argument("--model-name", type=str, default="traffic_sign_classifier")
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=args.url)

    input_data = preprocess(args.image)
    infer_input = httpclient.InferInput("input", input_data.shape, "FP32")
    infer_input.set_data_from_numpy(input_data)

    result = client.infer(model_name=args.model_name, inputs=[infer_input])
    logits = result.as_numpy("logits")[0]
    probs = softmax(logits)

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    print(f"predicted class: {pred_class}")
    print(f"confidence:      {confidence:.4f}")


if __name__ == "__main__":
    main()
