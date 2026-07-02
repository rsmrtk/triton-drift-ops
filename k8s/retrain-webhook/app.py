"""
Receives AlertManager webhooks for drift alerts and creates a Kubernetes
Job to retrain the model — this is what turns "drift detected" into
"retraining started" without a human in the loop.

Idempotent by design: AlertManager can and will re-send a firing alert
(group_interval), so before creating a Job this checks whether a retrain
Job is already running and skips if so, rather than piling up duplicate
training runs.

Runs inside the cluster with a ServiceAccount that has Job create/list
permissions in its namespace (see rbac.yaml) — it doesn't need a
kubeconfig, it uses the in-cluster config like any other pod talking to
the API server.
"""

import logging
import os
import time

from fastapi import FastAPI, Request
from kubernetes import client, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

NAMESPACE = os.getenv("NAMESPACE", "geo-mlops")
TRAIN_IMAGE = os.getenv("TRAIN_IMAGE", "ghcr.io/rsmrtk/triton-drift-ops-training:latest")
# cuda on a GPU cluster; cpu lets the same webhook run on a CPU-only
# cluster (k3d, CI) where a nvidia.com/gpu request would never schedule
TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "cuda")
TRAIN_EPOCHS = os.getenv("TRAIN_EPOCHS", "15")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# node path with a pre-downloaded GTSRB; when set, the Job mounts it
# read-only instead of downloading ~2GB into ephemeral storage per run
# (which is both slow and an eviction risk on small nodes)
TRAIN_DATA_HOSTPATH = os.getenv("TRAIN_DATA_HOSTPATH", "")
DATA_MOUNT = "/data/gtsrb-data"
RETRAIN_JOB_LABEL = "app=model-retrain"

app = FastAPI(title="retrain-webhook")

try:
    config.load_incluster_config()
except config.ConfigException:
    # allows running the webhook locally against a kubeconfig for testing
    config.load_kube_config()

batch_v1 = client.BatchV1Api()


def retrain_job_already_running() -> bool:
    jobs = batch_v1.list_namespaced_job(namespace=NAMESPACE, label_selector=RETRAIN_JOB_LABEL)
    for job in jobs.items:
        if job.status.active and job.status.active > 0:
            return True
    return False


def build_retrain_job_manifest() -> client.V1Job:
    job_name = f"model-retrain-{int(time.time())}"
    if TRAIN_DEVICE == "cuda":
        resources = client.V1ResourceRequirements(
            requests={"nvidia.com/gpu": "1", "memory": "4Gi", "cpu": "2"},
            limits={"nvidia.com/gpu": "1", "memory": "8Gi", "cpu": "4"},
        )
    else:
        resources = client.V1ResourceRequirements(
            requests={"memory": "2Gi", "cpu": "1"},
            limits={"memory": "4Gi", "cpu": "4"},
        )
    return client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name, labels={"app": "model-retrain"}),
        spec=client.V1JobSpec(
            ttl_seconds_after_finished=3600,
            backoff_limit=1,
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="trainer",
                            image=TRAIN_IMAGE,
                            command=["python", "train.py"],
                            args=[
                                "--epochs", TRAIN_EPOCHS,
                                "--device", TRAIN_DEVICE,
                                "--log-mlflow",
                                *(
                                    ["--data-dir", DATA_MOUNT]
                                    if TRAIN_DATA_HOSTPATH
                                    else []
                                ),
                            ],
                            env=[
                                client.V1EnvVar(
                                    name="MLFLOW_TRACKING_URI", value=MLFLOW_TRACKING_URI
                                ),
                            ],
                            resources=resources,
                            volume_mounts=(
                                [
                                    client.V1VolumeMount(
                                        name="gtsrb-data",
                                        mount_path=DATA_MOUNT,
                                        read_only=True,
                                    )
                                ]
                                if TRAIN_DATA_HOSTPATH
                                else None
                            ),
                        )
                    ],
                    volumes=(
                        [
                            client.V1Volume(
                                name="gtsrb-data",
                                host_path=client.V1HostPathVolumeSource(
                                    path=TRAIN_DATA_HOSTPATH, type="Directory"
                                ),
                            )
                        ]
                        if TRAIN_DATA_HOSTPATH
                        else None
                    ),
                )
            ),
        ),
    )


@app.post("/alert")
async def handle_alert(request: Request):
    payload = await request.json()
    alerts = payload.get("alerts", [])
    firing = [a for a in alerts if a.get("status") == "firing"]

    if not firing:
        logger.info("received webhook with no firing alerts, ignoring")
        return {"status": "ignored", "reason": "no firing alerts"}

    if retrain_job_already_running():
        logger.info("retrain Job already running, skipping (idempotent)")
        return {"status": "skipped", "reason": "retrain already in progress"}

    job = build_retrain_job_manifest()
    batch_v1.create_namespaced_job(namespace=NAMESPACE, body=job)
    logger.info("created retrain Job: %s", job.metadata.name)

    return {"status": "triggered", "job": job.metadata.name}


@app.get("/health")
def health():
    return {"status": "ok"}
