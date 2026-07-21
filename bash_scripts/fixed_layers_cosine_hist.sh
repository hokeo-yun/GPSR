#!/bin/bash

CUDA_ID="0"
TEST_DATA="GenImage"
DATASET_KEYS="stable_diffusion_v_1_4"
CKPT_PATH="./checkpoints/ptrain_3090_0_1/best_model.pth"
RESULT_FOLDER="./results/fixed_layers_cosine_hist/sdv1_4"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 analysis/fixed_layers_cosine_hist.py \
    --test_data="${TEST_DATA}" \
    --dataset_keys="${DATASET_KEYS}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=128 \
    --max_sample=5000 \
    --top_k_per_class=300 \
    --start_layer=13 \
    --end_layer=17 \
    --p=0.1
