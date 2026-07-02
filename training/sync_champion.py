"""
Bridge between the MLflow registry and Triton's model repository: makes
the current 'champion' alias the model Triton actually serves.

Runs as a CronJob next to Triton, sharing the model-repository PVC.
Each tick it looks up the champion version; if that version isn't in the
repository yet, it downloads the PyTorch weights from the registry,
exports ONNX and drops it in as a new numbered version. Triton runs with
--model-control-mode=poll, so the new version goes live within seconds —
no pod restart, which is the "promotion reaches serving without
downtime" arrow in the README diagram.

Idempotent: an already-synced champion is a no-op, so the schedule can
be tight (every minute) without churn.

Usage (in-cluster):
    MLFLOW_TRACKING_URI=http://mlflow:5000 python sync_champion.py
"""

import logging
import os
import sys
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "traffic-sign-classifier")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "traffic_sign_classifier")
MODEL_REPO = Path(os.getenv("MODEL_REPO", "/models/repo"))
# where the gateway looks for the drift baseline (its /baseline mount);
# the baseline travels with the champion so drift is always measured
# against the distribution of the model actually being served
BASELINE_DIR = Path(os.getenv("BASELINE_DIR", "/models/baseline"))
BASELINE_FILE = "baseline_confidences.npy"
# KIND_CPU or KIND_GPU — must match what the Triton pod can schedule on
TRITON_KIND = os.getenv("TRITON_KIND", "KIND_CPU")
ALIAS = "champion"
INPUT_SIZE = 48
NUM_CLASSES = 43

CONFIG_PBTXT = f"""name: "{TRITON_MODEL_NAME}"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, {INPUT_SIZE}, {INPUT_SIZE} ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ {NUM_CLASSES} ]
  }}
]

instance_group [
  {{
    count: 1
    kind: {TRITON_KIND}
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 4, 8, 16 ]
  max_queue_delay_microseconds: 5000
}}
"""


def main() -> int:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        logger.info("no '%s' alias on '%s' yet — nothing to sync", ALIAS, MODEL_NAME)
        return 0

    model_dir = MODEL_REPO / TRITON_MODEL_NAME
    version_dir = model_dir / champion.version
    if (version_dir / "model.onnx").exists():
        logger.info("champion v%s already in the model repository — no-op", champion.version)
        return 0

    logger.info("syncing champion v%s (run %s) into %s", champion.version, champion.run_id, version_dir)
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@{ALIAS}", map_location="cpu")
    model.eval()

    model_dir.mkdir(parents=True, exist_ok=True)
    config = model_dir / "config.pbtxt"
    if not config.exists():
        config.write_text(CONFIG_PBTXT)
        logger.info("wrote %s (%s)", config, TRITON_KIND)

    version_dir.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(version_dir / "model.onnx"),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        # see serving/export_onnx.py — the torch>=2.9 dynamo exporter pins
        # the output shape and Triton rejects it; the legacy path is correct
        dynamo=False,
    )
    logger.info(
        "champion v%s exported — Triton's poll loop will load it within seconds",
        champion.version,
    )

    try:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = client.download_artifacts(champion.run_id, BASELINE_FILE, str(BASELINE_DIR))
        logger.info("baseline distribution synced to %s", downloaded)
    except MlflowException as e:
        # a champion without a logged baseline still serves — the gateway
        # just reports no drift, same degradation stance as everywhere else
        logger.warning("no %s artifact on champion run: %s", BASELINE_FILE, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
