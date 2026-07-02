"""
Measure how much a trained model's accuracy degrades under each drift
scenario, relative to the clean baseline. This is the offline counterpart
to the online drift monitor (stage 3b) — run it once after training to
know what "drift" actually costs in accuracy before wiring up live
detection.

Usage:
    python evaluate_drift.py --model ../training/model.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from data.dataset import get_loaders  # noqa: E402
from model.net import TrafficSignNet  # noqa: E402
from transforms import DRIFT_SCENARIOS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="../training/gtsrb-data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="drift_report.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = TrafficSignNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    results: dict[str, float] = {}
    for name, transform in DRIFT_SCENARIOS.items():
        logger.info("evaluating scenario: %s", name)
        _, test_loader = get_loaders(data_dir=args.data_dir, extra_transform=transform)
        accuracy = evaluate(model, test_loader, device)
        results[name] = round(accuracy, 4)
        logger.info("  accuracy: %.4f", accuracy)

    baseline = results["clean"]
    logger.info("\n--- summary (relative to clean baseline: %.4f) ---", baseline)
    for name, acc in results.items():
        delta = acc - baseline
        logger.info("  %-14s %.4f  (%+.4f)", name, acc, delta)

    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
