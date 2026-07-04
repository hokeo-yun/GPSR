#!/bin/bash


# CUDA_ID="0"
# CKPT_PATH="./reproduce_checkpoints/reproduce/best_model.pth"
# RESULT_FOLDER="results/"

# CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
#     --arch="CLIP:ViT-L/14" \
#     --ckpt="${CKPT_PATH}" \
#     --result_folder="${RESULT_FOLDER}" \
#     --select_k=5 \
#     --batch_size=256


# CUDA_ID="0"
# CKPT_PATH="./reproduce_checkpoints/PPLT/best_model.pth"
# RESULT_FOLDER="results/"
# TEST_DATA="UFD"

# CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
#     --arch="CLIP:ViT-L/14" \
#     --ckpt="${CKPT_PATH}" \
#     --result_folder="${RESULT_FOLDER}" \
#     --select_k=5 \
#     --batch_size=256

CUDA_ID="0"
CKPT_PATH="./checkpoints/p/p6/best_model.pth"
RESULT_FOLDER="./results/p/p6/"

TEST_DATA="GenImage"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=1 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256