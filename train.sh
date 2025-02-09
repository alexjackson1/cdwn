#!/bin/bash

set -euxo pipefail

BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
DATA_DIR=/workspace/data/countdown
PROJECT_NAME=cdwn
EXPERIMENT_NAME=cdwn-2.5-0.5B-Instruct-6-gr
ROLLOUT_TP_SIZE=1
NUM_GPUS=1
DATASET_NAME="alexjackson17/countdown-numbers-6-gr"

python3 examples/data_preprocess/countdown.py --local_dir $DATA_DIR --dataset $DATASET_NAME --template_type qwen-instruct

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$DATA_DIR/train.parquet \
 data.val_files=$DATA_DIR/test.parquet \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$EXPERIMENT_NAME \
 data.train_batch_size=256 \
 data.val_batch_size=1312 \
 data.max_prompt_length=256 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=$BASE_MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.ppo_micro_batch_size=8 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=$BASE_MODEL \
 critic.ppo_micro_batch_size=8 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 +trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$NUM_GPUS \
 trainer.nnodes=1 \
 trainer.save_freq=100 \
 trainer.test_freq=100 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log
