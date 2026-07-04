#!/bin/bash

CUDA_ID="1"
TEST_DATA="GenImage"

for i in {5..7}
do
    SETTING="setting${i}_3090"

    CKPT_PATH="./checkpoints/${SETTING}/best_model.pth"
    RESULT_FOLDER="./results/${SETTING}/"

    echo "======================================"
    echo "Testing ${SETTING}"
    echo "CKPT: ${CKPT_PATH}"
    echo "Result: ${RESULT_FOLDER}"
    echo "======================================"

    CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
        --p=1 \
        --test_data="${TEST_DATA}" \
        --arch="CLIP:ViT-L/14" \
        --ckpt="${CKPT_PATH}" \
        --result_folder="${RESULT_FOLDER}" \
        --select_k=5 \
        --batch_size=256
done

CKPT_PATH="./checkpoints/default_setting_3090/best_model.pth"
RESULT_FOLDER="./results/default_setting_3090/"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=1 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256

