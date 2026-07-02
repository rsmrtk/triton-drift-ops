# Stage 8 — the real GPU run

Everything before this stage runs CPU-only. This is the one-time,
credit-bounded GPU session that proves the same code and images work on
real NVIDIA hardware, and produces the demo material (video/screenshots
of the full drift → retrain → promote loop).

Budget note: a single `g2-standard-4` (1× L4) on GCP costs roughly
$0.85/hr. The whole session below fits in 2–3 hours — a few dollars of
free-trial credit. Create the VM, do the run, capture the demo, **delete
the VM the same day**.

## 1. VM with GPU (GCP free trial)

```bash
gcloud compute instances create triton-drift-gpu \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu124-ubuntu-2204 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

The Deep Learning VM image ships with the NVIDIA driver and Docker +
NVIDIA Container Toolkit preinstalled — no manual driver installs.

New GCP accounts may need a GPU quota bump (`GPUS_ALL_REGIONS` from 0 to
1) — request it first, approval is usually quick.

## 2. Smoke test (stage 0 finally runs for real)

```bash
git clone https://github.com/rsmrtk/triton-drift-ops && cd triton-drift-ops
docker build -t smoke docker/nvidia-smoke-test
docker run --rm --gpus all smoke   # expect the nvidia-smi table with the L4
```

Screenshot this — it's the "GPU passthrough works" proof point.

## 3. GPU training run

```bash
docker run --rm --gpus all \
  -v $PWD/out:/app/out \
  ghcr.io/rsmrtk/triton-drift-ops-training:latest \
  python train.py --epochs 15 --device cuda --out /app/out/model.pt
```

Compare the per-epoch wall time against the CPU numbers in the README —
the speedup is part of the story.

## 4. Serve through Triton on GPU

```bash
# flip the instance group to GPU for this run
sed -i 's/KIND_CPU/KIND_GPU/' serving/model_repository/traffic_sign_classifier/config.pbtxt

python serving/export_onnx.py --model out/model.pt \
  --out serving/model_repository/traffic_sign_classifier/1/model.onnx

cd serving && docker compose up   # add --gpus all / device reservation for triton
```

Then hit the gateway with test images (`serving/client.py` or curl) and
screenshot Triton's startup log showing the model loaded on GPU.

## 5. The demo loop (the money shot)

1. Send a stream of clean GTSRB test images through the gateway —
   Grafana shows `model_drift_score` near 0.
2. Switch the stream to a drift scenario (`drift/transforms.py`, e.g.
   `night`) — watch `model_drift_score` climb past 0.25.
3. Alert fires → retrain-webhook creates the training Job → new MLflow
   version appears → promotion gate compares accuracies.
4. Record the whole thing as one screen capture: drift graph rising,
   alert firing, Job starting, new champion promoted.

## 6. Tear down

```bash
gcloud compute instances delete triton-drift-gpu --zone=us-central1-a
```

Same day. The artifacts (model, screenshots, video, MLflow export) are
the deliverable — the VM is disposable.
