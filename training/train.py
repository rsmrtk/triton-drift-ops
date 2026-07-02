"""
Baseline training run for the traffic sign classifier.

CPU-friendly by design — this is meant to run during development without
GPU credits. The same script runs unchanged on GPU (stage 8) via
`--device cuda`, which is the point: the training code doesn't change
between dev and the real GPU run, only the flag.

Usage:
    python train.py --epochs 5 --device cpu
    python train.py --epochs 15 --device cuda --drift-scenario night
    MLFLOW_TRACKING_URI=http://localhost:5000 python train.py --epochs 15 --log-mlflow
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import get_loaders
from model.net import TrafficSignNet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "drift"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "traffic-sign-classifier")
EXPERIMENT_NAME = "triton-drift-ops"


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def collect_confidences(model, loader, device) -> "np.ndarray":
    """
    Softmax confidence of the predicted class for every test sample.
    This is the training-time baseline the online drift monitor compares
    live traffic against (drift/monitor.py) — saved next to the weights
    and shipped to the inference gateway.
    """
    import numpy as np

    model.eval()
    confidences: list[float] = []
    for images, _ in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        confidences.extend(probs.max(dim=1).values.cpu().tolist())
    return np.array(confidences, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./gtsrb-data")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=str, default="./model.pt")
    parser.add_argument(
        "--drift-scenario",
        type=str,
        default="clean",
        help="train on a drifted distribution instead of clean data — used to produce a "
        "retraining candidate after drift has been detected (see drift/transforms.py)",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="log params/metrics/model to MLflow and register a new model version",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        logger.warning("cuda requested but not available — falling back to cpu")

    extra_transform = None
    if args.drift_scenario != "clean":
        from transforms import get_drift_transform  # noqa: E402

        extra_transform = get_drift_transform(args.drift_scenario)
        logger.info("training on drift scenario: %s", args.drift_scenario)

    logger.info("loading GTSRB...")
    train_loader, test_loader = get_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, extra_transform=extra_transform
    )

    model = TrafficSignNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    final_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        final_accuracy = evaluate(model, test_loader, device)
        elapsed = time.time() - start
        logger.info(
            "epoch %d/%d — loss: %.4f, test accuracy: %.4f (%.1fs)",
            epoch, args.epochs, train_loss, final_accuracy, elapsed,
        )

    torch.save(model.state_dict(), args.out)
    logger.info("saved model weights to %s", args.out)

    import numpy as np

    baseline = collect_confidences(model, test_loader, device)
    baseline_path = Path(args.out).with_name("baseline_confidences.npy")
    np.save(baseline_path, baseline)
    logger.info(
        "saved %d baseline confidences to %s (mean %.4f)",
        len(baseline), baseline_path, baseline.mean(),
    )

    if args.log_mlflow:
        log_to_mlflow(model, args, final_accuracy, baseline_path)


def log_to_mlflow(
    model: nn.Module, args: argparse.Namespace, accuracy: float, baseline_path: Path
) -> None:
    import mlflow
    import mlflow.pytorch

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "drift_scenario": args.drift_scenario,
                "device": args.device,
            }
        )
        mlflow.log_metric("accuracy", accuracy)
        # baseline distribution travels with the model version so the
        # gateway always compares live traffic against the right baseline
        mlflow.log_artifact(str(baseline_path))
        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

    logger.info("logged run to MLflow, registered as '%s'", MODEL_NAME)


if __name__ == "__main__":
    main()
