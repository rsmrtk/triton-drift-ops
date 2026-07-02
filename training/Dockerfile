# Build from the repo root so drift/ (drift-scenario training) comes along:
#   docker build -f training/Dockerfile -t triton-drift-ops-training .
#
# python:slim + pip torch is deliberate: the Linux torch wheel bundles its
# own CUDA libraries, so this single image trains on CPU (dev) and on GPU
# (via NVIDIA Container Toolkit / device plugin) with no Dockerfile change —
# mirroring how train.py itself only differs by --device.
FROM python:3.12-slim

WORKDIR /app
COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ .
COPY drift/transforms.py ../drift/transforms.py

CMD ["python", "train.py", "--epochs", "15", "--device", "cuda", "--log-mlflow"]
