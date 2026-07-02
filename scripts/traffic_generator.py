"""
Demo traffic generator: streams GTSRB test images through the inference
gateway, optionally degraded by a drift scenario. This is how the demo
moves the drift needle — start with `--scenario clean`, watch
model_drift_score sit near zero, then restart with `--scenario night`
and watch it climb until the retraining alert fires.

Usage:
    python traffic_generator.py --gateway http://localhost:8081 --scenario clean
    python traffic_generator.py --gateway http://localhost:8081 --scenario night --rps 5
"""

import argparse
import io
import logging
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "drift"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from transforms import DRIFT_SCENARIOS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def iter_test_images(data_dir: str):
    """Yields PIL images from the extracted GTSRB test split, looping forever."""
    from PIL import Image

    root = Path(data_dir)
    paths = sorted(root.rglob("*.ppm")) + sorted(root.rglob("*.png"))
    if not paths:
        raise SystemExit(
            f"no images under {root} — run training once first (it downloads GTSRB), "
            "or point --data-dir at any folder of images"
        )
    logger.info("found %d images", len(paths))
    while True:
        yield Image.open(random.choice(paths)).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", type=str, default="http://localhost:8081")
    parser.add_argument("--scenario", type=str, default="clean", choices=list(DRIFT_SCENARIOS))
    parser.add_argument("--rps", type=float, default=2.0, help="requests per second")
    parser.add_argument("--data-dir", type=str, default="../training/gtsrb-data")
    parser.add_argument("--count", type=int, default=0, help="stop after N requests (0 = run forever)")
    args = parser.parse_args()

    transform = DRIFT_SCENARIOS[args.scenario]
    delay = 1.0 / args.rps
    logger.info("streaming '%s' traffic to %s at %.1f rps", args.scenario, args.gateway, args.rps)

    sent = errors = 0
    for img in iter_test_images(args.data_dir):
        for t in transform.transforms:
            img = t(img)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        try:
            resp = requests.post(
                f"{args.gateway}/classify",
                files={"file": ("sign.png", buf, "image/png")},
                timeout=5,
            )
            resp.raise_for_status()
            body = resp.json()
            sent += 1
            if sent % 25 == 0:
                logger.info(
                    "%d sent — last: class=%s confidence=%.4f",
                    sent, body["class_id"], body["confidence"],
                )
        except requests.RequestException as e:
            errors += 1
            logger.warning("request failed (%d errors so far): %s", errors, e)

        if args.count and sent >= args.count:
            logger.info("done: %d sent, %d errors", sent, errors)
            return
        time.sleep(delay)


if __name__ == "__main__":
    main()
