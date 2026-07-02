"""
Promote a model version to the "champion" alias — but only if it actually
beats the currently-promoted version's accuracy. This is the promotion
gate: an unconditional promote (like a naive "always ship the latest
version") would let a retraining run that produced a worse model overwrite
a good one. Comparing against the current champion before promoting is
what makes the retraining loop safe to run unattended.

Usage:
    python promote.py                    # compare latest version vs current champion
    python promote.py --version 5        # compare a specific version instead
    python promote.py --force            # skip the comparison, promote anyway
"""

import argparse
import logging
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "traffic-sign-classifier")
ALIAS = "champion"


def get_run_accuracy(client: MlflowClient, run_id: str) -> float | None:
    run = client.get_run(run_id)
    return run.data.metrics.get("accuracy")


def promote(version: int | None, force: bool) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise RuntimeError(f"no versions found for '{MODEL_NAME}'")
        version = max(int(v.version) for v in versions)
        logger.info("latest version: %d", version)

    candidate = client.get_model_version(MODEL_NAME, str(version))
    candidate_accuracy = get_run_accuracy(client, candidate.run_id)
    if candidate_accuracy is None:
        raise RuntimeError(f"version {version} has no 'accuracy' metric logged — cannot evaluate promotion gate")

    if force:
        logger.info("--force set, skipping comparison against current champion")
    else:
        try:
            current_champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
            champion_accuracy = get_run_accuracy(client, current_champion.run_id)
        except MlflowException:
            champion_accuracy = None

        if champion_accuracy is None:
            logger.info("no current champion — promoting version %d unconditionally", version)
        elif candidate_accuracy <= champion_accuracy:
            logger.info(
                "version %d (accuracy %.4f) does not beat current champion (accuracy %.4f) — not promoting",
                version, candidate_accuracy, champion_accuracy,
            )
            return
        else:
            logger.info(
                "version %d (accuracy %.4f) beats current champion (accuracy %.4f)",
                version, candidate_accuracy, champion_accuracy,
            )

    client.set_registered_model_alias(name=MODEL_NAME, alias=ALIAS, version=str(version))
    logger.info("version %d promoted to '%s' (accuracy %.4f)", version, ALIAS, candidate_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="promote without comparing accuracy")
    args = parser.parse_args()
    promote(args.version, args.force)
