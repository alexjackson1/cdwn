#!/bin/bash

set -euxo pipefail

export DATA_DIR="/workspace/data/countdown"
export DATASET_NAME="alexjackson17/countdown-numbers-3-8-nz"

python3 examples/data_preprocess/countdown.py \
  --local_dir $DATA_DIR \
  --dataset $DATASET_NAME \
  --template_type base \
  --perfect_solutions
