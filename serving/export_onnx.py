"""
Export a trained TrafficSignNet checkpoint to ONNX for Triton serving.

Triton's ONNX Runtime backend is used here rather than the native PyTorch
backend — ONNX is the more portable choice (works with the ONNX Runtime
CPU or GPU execution provider without requiring libtorch in the Triton
container) and is the more common path for a first Triton deployment.

Usage:
    python export_onnx.py --model ../training/model.pt --out model_repository/traffic_sign_classifier/1/model.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from model.net import INPUT_SIZE, TrafficSignNet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to .pt state_dict")
    parser.add_argument("--out", type=str, required=True, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = TrafficSignNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=args.opset,
    )
    print(f"exported ONNX model to {out_path}")


if __name__ == "__main__":
    main()
