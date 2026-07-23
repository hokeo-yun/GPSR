#!/bin/bash

CUDA_ID="0"
TEST_DATA="ForenSynths"
DATASET_KEYS="progan"
CKPT_PATH="./checkpoints/ptrain_3090_0_1/best_model.pth"
RESULT_FOLDER="./results/all_layers_ps_cosine/progan"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 analysis/all_layers_ps_cosine.py \
    --test_data="${TEST_DATA}" \
    --dataset_keys="${DATASET_KEYS}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=128 \
    --max_sample=2000 \
    --top_k_per_class=500 \
    --p=0.1