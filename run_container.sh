#!/bin/bash

set -euxo pipefail

docker run --runtime=nvidia \
  -it --rm --shm-size="10g" \
  --cap-add=SYS_ADMIN \
  --volume ./run:/workspace/run \
  --env WANDB_API_KEY=$WANDB_API_KEY \
  --entrypoint "/workspace/run/preprocess-base.sh && /workspace/run/train.sh" \
  ghcr.io/alexjackson1/cdwn:latest

