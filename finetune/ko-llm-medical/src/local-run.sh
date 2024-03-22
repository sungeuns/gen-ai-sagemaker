#!/bin/bash

set -e
pip install -r requirements.txt

mkdir -p /tmp/huggingface-cache/
export HF_DATASETS_CACHE="/tmp/huggingface-cache"

BASE_MODEL_PATH="/home/ec2-user/SageMaker/models/LDCC-SOLAR-10-7B/models--LDCC--LDCC-SOLAR-10.7B/snapshots/1055563879363d9ee2fba1d9fd1628eca6bcbb4e"
DATA_PATH="../debug_dataset"
OUTPUT_PATH="../local_output"
MERGED_MODEL_PATH="../local_merged"


declare -a OPTS=(
    --pretrained_model_path $BASE_MODEL_PATH
    --cache_dir $HF_DATASETS_CACHE
    --data_path $DATA_PATH
    --output_dir $OUTPUT_PATH
    --save_path $MERGED_MODEL_PATH
    --batch_size 2
    --gradient_accumulation_steps 2
    --num_epochs 1
    --learning_rate 3e-4
    --lora_r 8
    --lora_alpha 16
    --lora_dropout 0.05
    --logging_steps 1
    --save_steps 200
    --eval_steps 200
    --weight_decay 0.
    --warmup_steps 50
    --warmup_ratio 0.03
    --lr_scheduler_type "linear"
)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $NUM_GPUS -eq 1 ]
then
    echo python train.py "${OPTS[@]}" "$@"
    CUDA_VISIBLE_DEVICES=0 python train.py "${OPTS[@]}" "$@"
else
    echo torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" train.py "${OPTS[@]}" "$@"
    torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" train.py "${OPTS[@]}" "$@"
fi