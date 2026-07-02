#!/usr/bin/env bash
# Stage 8: create the one-shot GPU demo VM. Costs ~$0.85/hr — delete it
# the same day with scripts/gcp/delete_vm.sh.
#
# Prereqs (one-time, per project):
#   - GPUS_ALL_REGIONS quota >= 1 (Cloud Quotas request, usually instant)
#   - a grafana-admin-password secret in Secret Manager:
#       python3 -c 'import secrets; print(secrets.token_urlsafe(18))' \
#         | gcloud secrets create grafana-admin-password --data-file=- --project "$PROJECT"
set -euo pipefail

PROJECT="${PROJECT:-triton-drift}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-triton-drift-gpu}"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True \
  --scopes=cloud-platform

echo
echo "SSH:      gcloud compute ssh $NAME --project=$PROJECT --zone=$ZONE"
echo "Teardown: PROJECT=$PROJECT ZONE=$ZONE NAME=$NAME scripts/gcp/delete_vm.sh"
