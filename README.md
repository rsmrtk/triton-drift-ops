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
                         ┌─────────────────────┐
  traffic-sign images ──▶│  Triton Inference    │──▶ prediction + confidence
                         │  Server (GPU)         │
                         └──────────┬───────────┘
                                    │ prediction logs
                                    ▼
                         ┌─────────────────────┐
                         │  Drift monitor        │  compares live confidence /
                         │  (Prometheus metrics)  │  embedding distribution vs.
                         └──────────┬───────────┘  training baseline
                                    │ drift score > threshold
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
- [ ] Stage 0 — Docker + NVIDIA Container Toolkit smoke test (`docker/nvidia-smoke-test`)
- [ ] Stage 1 — Baseline training run on GTSRB (clean data, CPU-friendly for dev)
- [ ] Stage 2 — MLflow tracking + model registry integration
- [ ] Stage 3 — Drift simulation (fog/night/noise transforms) + drift metrics
- [ ] Stage 4 — Auto-retraining trigger (Prometheus alert → K8s Job)
- [ ] Stage 5 — NVIDIA Triton serving + model repository layout
- [ ] Stage 6 — Promotion gate (new model must beat champion to be promoted)
- [ ] Stage 7 — Helm charts + ArgoCD Application manifests
- [ ] Stage 8 — End-to-end GPU run (cloud trial credits) + demo capture

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
