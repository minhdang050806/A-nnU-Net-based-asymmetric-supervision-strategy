#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.synapse.org/syn74549440/lisa-os50-aura:v1}"
docker push "${IMAGE}"
