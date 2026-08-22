#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-docker.synapse.org/syn74549440/lisa-os50-aura:v1}"
INPUT_DIR="${2:?Usage: ./test.sh [IMAGE] INPUT_DIR [OUTPUT_DIR]}"
OUTPUT_DIR="${3:-${PWD}/docker_test_output}"
mkdir -p "${OUTPUT_DIR}"

docker run --rm --network none --gpus all \
  --volume "$(realpath "${INPUT_DIR}"):/input:ro" \
  --volume "$(realpath "${OUTPUT_DIR}"):/output:rw" \
  "${IMAGE}"
