#!/bin/bash

CUDA_ID="0"
TEST_DATA="GenImage"
DATASET_KEYS="ADM,Midjourney,stable_diffusion_v_1_4"
CKPT_PATH="./checkpoints/ptrain_3090_0_1/best_model.pth"
RESULT_FOLDER="./results/ps_delta_analysis"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 analysis/ps_delta_analysis.py \
    --test_data="${TEST_DATA}" \
    --dataset_keys="${DATASET_KEYS}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=128 \
    --max_sample=300 \
    --mode="ps" \
    --p=0.1
