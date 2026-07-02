"""
Online drift monitor: watches live prediction confidence scores and flags
when their distribution has shifted meaningfully from the training-time
baseline. Exposes the result as Prometheus metrics so an AlertManager rule
can trigger retraining without a human watching a dashboard.

Design choice: confidence-score distribution rather than accuracy, because
in production you don't have ground-truth labels for incoming traffic —
you only have what the model predicted and how sure it was. A model that's
seen drifted input tends to produce a *lower and more spread out*
confidence distribution even when you can't check if predictions are
"correct" — this is the same signal papers call "predictive entropy" or
"softmax response" drift detection.

Usage (as a library, called by the serving layer after each batch of
predictions):

    from drift.monitor import DriftMonitor
    monitor = DriftMonitor(baseline_confidences=baseline)
    monitor.observe(confidence_scores)   # call after every inference batch
    # exposes drift_score, mean_confidence, low_confidence_ratio via
    # prometheus_client on :9100/metrics
"""

import logging
from collections import deque

import numpy as np
from prometheus_client import Gauge, start_http_server
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DRIFT_SCORE = Gauge(
    "model_drift_score",
    "KS-statistic distance between live and baseline confidence distributions (0=identical, 1=fully shifted)",
)
MEAN_CONFIDENCE = Gauge(
    "model_mean_confidence",
    "Rolling mean of prediction confidence over the current window",
)
LOW_CONFIDENCE_RATIO = Gauge(
    "model_low_confidence_ratio",
    "Fraction of recent predictions with confidence below 0.5",
)
WINDOW_SIZE_METRIC = Gauge(
    "model_drift_window_size",
    "Number of predictions currently in the rolling window",
)

DEFAULT_WINDOW_SIZE = 500
LOW_CONFIDENCE_THRESHOLD = 0.5


class DriftMonitor:
    """
    Rolling-window drift detector over prediction confidence scores.

    drift_score is the two-sample Kolmogorov-Smirnov statistic between the
    live window and the baseline (training-time) confidence distribution —
    a standard, dependency-light choice for comparing two 1D distributions
    without assuming they're normal.
    """

    def __init__(
        self,
        baseline_confidences: np.ndarray,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ):
        if len(baseline_confidences) < 30:
            raise ValueError("baseline_confidences needs at least 30 samples for a meaningful KS test")
        self.baseline = np.asarray(baseline_confidences)
        self.window: deque[float] = deque(maxlen=window_size)

    def observe(self, confidences: np.ndarray) -> dict[str, float]:
        self.window.extend(float(c) for c in confidences)
        return self._compute_metrics()

    def _compute_metrics(self) -> dict[str, float]:
        if len(self.window) < 30:
            return {"drift_score": 0.0, "mean_confidence": 0.0, "low_confidence_ratio": 0.0}

        live = np.array(self.window)
        ks_result = stats.ks_2samp(live, self.baseline)
        drift_score = float(ks_result.statistic)
        mean_confidence = float(live.mean())
        low_confidence_ratio = float((live < LOW_CONFIDENCE_THRESHOLD).mean())

        DRIFT_SCORE.set(drift_score)
        MEAN_CONFIDENCE.set(mean_confidence)
        LOW_CONFIDENCE_RATIO.set(low_confidence_ratio)
        WINDOW_SIZE_METRIC.set(len(self.window))

        return {
            "drift_score": drift_score,
            "mean_confidence": mean_confidence,
            "low_confidence_ratio": low_confidence_ratio,
        }


def serve_metrics(port: int = 9100) -> None:
    start_http_server(port)
    logger.info("drift metrics exposed on :%d/metrics", port)
