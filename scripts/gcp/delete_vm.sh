#!/usr/bin/env bash
# Stage 8 teardown: the VM is disposable, the artifacts (model, metrics,
# demo capture) are the deliverable. Run this the same day as create_vm.sh.
set -euo pipefail

PROJECT="${PROJECT:-triton-drift}"
ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-triton-drift-gpu}"

gcloud compute instances delete "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet
echo "deleted $NAME — remaining billed resources in $PROJECT: none expected"
