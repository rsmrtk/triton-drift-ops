# triton-drift-ops

A reference MLOps project: a vision classifier (traffic sign recognition,
[GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html)) deployed with NVIDIA
Triton Inference Server, monitored for data drift in production, and
automatically retrained and promoted when accuracy degrades — no manual
intervention after the pipeline is wired up.

The point of this project is the operational loop, not the model. The
classifier is intentionally simple (a small CNN/ResNet); the engineering
value is in detecting drift, triggering retraining safely, and promoting a
new model version without downtime.

## Why this project

Most MLOps demos stop at "trained a model, deployed it." The harder and more
realistic problem is what happens *after* deployment: the input distribution
shifts (worse weather, different camera, degraded image quality), model
accuracy silently drops, and someone has to notice and act. This project
builds that detection + reaction loop end to end, and serves the model
through NVIDIA's production inference stack rather than a hand-rolled
FastAPI wrapper.

## Architecture

```
                         ┌─────────────────────┐      ┌─────────────────────┐
  traffic-sign images ──▶│  Inference gateway   │─────▶│  Triton Inference    │
                         │  (FastAPI)            │◀─────│  Server (CPU/GPU)     │
                         └──────────┬───────────┘      └─────────────────────┘
                                    │ softmax confidence, every request
                                    ▼
                         ┌─────────────────────┐
                         │  Drift monitor        │  KS-test: live confidence
                         │  (Prometheus metrics)  │  distribution vs. training
                         └──────────┬───────────┘  baseline
                                    │ drift score > threshold, sustained
                                    ▼
                         ┌─────────────────────┐
                         │  Retraining trigger   │  Prometheus AlertManager
                         │  (K8s Job)             │  → webhook → Job
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Training (GPU)        │  logs metrics + model to
                         │  → MLflow Registry     │  MLflow, registers new version
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Promotion gate        │  new version only promoted to
                         │  (accuracy comparison) │  "champion" alias if it beats
                         └──────────┬───────────┘  the currently served version
                                    ▼
                         Triton reloads champion model (no restart)
```

GitOps deploy path (Helm + ArgoCD) mirrors the split used in
[geo-mlops-infra](https://github.com/rsmrtk/geo-mlops-infra): app/training
code in this repo, cluster state reconciled from Git.

## Status

Built incrementally, in order — each stage only starts once the previous
one is proven working:

- [x] Repo scaffolding
- [x] Stage 0 — Docker + NVIDIA Container Toolkit smoke test (`docker/nvidia-smoke-test`) — written, not yet run on real GPU hardware
- [x] Stage 1 — Baseline training script on GTSRB (`training/train.py`) — written, syntax-checked; not yet run end to end (blocked locally by disk space, see below)
- [x] Stage 2 — MLflow tracking + model registry integration (`--log-mlflow` flag in `train.py`, `training/promote.py` with a promotion gate) — written, not yet run against a live MLflow server
- [x] Stage 3 — Drift simulation (fog/night/noise/motion-blur transforms, `drift/transforms.py`) + offline drift evaluation (`drift/evaluate_drift.py`) + online drift monitor with Prometheus metrics (`drift/monitor.py`) — written, not yet run
- [x] Stage 4 — Auto-retraining trigger: PrometheusRule on drift metrics (`k8s/alerts/drift-alert-rule.yaml`), AlertManager routing (`k8s/alerts/alertmanager-config.yaml`), idempotent webhook that creates a K8s training Job (`k8s/retrain-webhook`) — written, not yet run
- [x] Stage 5 — NVIDIA Triton model repository + config (`serving/model_repository`), ONNX export (`serving/export_onnx.py`), HTTP client (`serving/client.py`), inference gateway that feeds live confidences to the drift monitor (`serving/gateway`), local `docker-compose.yaml` — written, not yet run
- [x] Stage 6 — Promotion gate (`training/promote.py` only promotes a version if its logged accuracy beats the current champion's)
- [x] Stage 7 — Helm chart for the whole stack with a CPU/GPU toggle (`helm/triton-drift-ops`), ArgoCD Application (`argocd/app.yaml`), CI building all three images (`.github/workflows/ci.yaml`) — lints and renders, not yet deployed
- [ ] Stage 8 — End-to-end GPU run (cloud trial credits) + demo capture

**Current limitation:** the full pipeline is written but not yet verified
end to end — local development ran out of disk space installing PyTorch,
so no training run, MLflow logging, or Triton serving has actually been
executed yet. Next concrete step is running `training/train.py` in a
container (not a local venv) to validate the training loop before wiring
up the rest.

## Stack

PyTorch · NVIDIA Triton Inference Server · NVIDIA Container Toolkit ·
MLflow · Prometheus/Grafana · Kubernetes · Helm · ArgoCD · Docker

## Stage 0 — GPU smoke test

Before building anything else, confirm the container runtime can actually
see a GPU:

```bash
cd docker/nvidia-smoke-test
docker build -t triton-drift-ops/nvidia-smoke-test .
docker run --rm --gpus all triton-drift-ops/nvidia-smoke-test
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on the host. On a machine without a GPU, this is expected to fail —
the rest of the pipeline (training, drift detection) is developed CPU-only
and only needs a real GPU for the final training/serving stages.
