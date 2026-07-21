#!/bin/bash

CUDA_ID="0"
TEST_DATA="ForenSynths"
CKPT_PATH="./checkpoints/ptrain_3090_0_1/best_model.pth"
RESULT_ROOT="./results/robustness_jpeg_blur_ForenSynths"

ARCH="CLIP:ViT-L/14"
SELECT_K=5
BATCH_SIZE=256
P=0.1

echo "======================================"
echo "Clean baseline"
echo "CKPT: ${CKPT_PATH}"
echo "Result: ${RESULT_ROOT}/clean/"
echo "======================================"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p="${P}" \
    --test_data="${TEST_DATA}" \
    --arch="${ARCH}" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_ROOT}/clean/" \
    --select_k="${SELECT_K}" \
    --batch_size="${BATCH_SIZE}"

for QUALITY in 100 90 80 70 60
do
    echo "======================================"
    echo "JPEG robustness: quality=${QUALITY}"
    echo "CKPT: ${CKPT_PATH}"
    echo "Result: ${RESULT_ROOT}/jpeg_q${QUALITY}/"
    echo "======================================"

    CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
        --p="${P}" \
        --test_data="${TEST_DATA}" \
        --arch="${ARCH}" \
        --ckpt="${CKPT_PATH}" \
        --result_folder="${RESULT_ROOT}/jpeg_q${QUALITY}/" \
        --select_k="${SELECT_K}" \
        --batch_size="${BATCH_SIZE}" \
        --jpeg_quality="${QUALITY}"
done

for SIGMA in 0 1 2 3 4
do
    echo "======================================"
    echo "Gaussian blur robustness: sigma=${SIGMA}"
    echo "CKPT: ${CKPT_PATH}"
    echo "Result: ${RESULT_ROOT}/blur_sigma${SIGMA}/"
    echo "======================================"

    CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
        --p="${P}" \
        --test_data="${TEST_DATA}" \
        --arch="${ARCH}" \
        --ckpt="${CKPT_PATH}" \
        --result_folder="${RESULT_ROOT}/blur_sigma${SIGMA}/" \
        --select_k="${SELECT_K}" \
        --batch_size="${BATCH_SIZE}" \
        --gaussian_sigma="${SIGMA}"
done
