#!/bin/bash

set -e

TOTAL_EPOCHS=5
TRAIN_CUDA=1
CUDA_ID=1

CHECKPOINTS_DIR="checkpoints/p_trainp"
RESULT_ROOT="./results/p_trainp"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"

ARCH="CLIP:ViT-L/14"
LR=0.00005
SELECT_K=5
BATCH_SIZE=256

P_LIST=(0.6 0.7 0.8 0.9 1.0)
TEST_LIST=("UFD" "GenImage")

for P in "${P_LIST[@]}"; do
    P_TAG=$(echo "${P}" | sed 's/\./_/g')
    NAME="setting_3090_p${P_TAG}"

    echo "=========================================="
    echo "Training with p=${P}"
    echo "Experiment name: ${NAME}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
        --p="${P}" \
        --name="${NAME}" \
        --wang2020_data_path="${DATA_DIR}" \
        --checkpoints_dir="${CHECKPOINTS_DIR}" \
        --data_mode="sd1_4" \
        --arch="${ARCH}" \
        --lr="${LR}" \
        --fix_backbone \
        --select_k="${SELECT_K}" \
        --batch_size="${BATCH_SIZE}" \
        --save_epoch_freq=5 \
        --niter="${TOTAL_EPOCHS}"

    CKPT_PATH="./${CHECKPOINTS_DIR}/${NAME}/best_model.pth"
    RESULT_FOLDER="${RESULT_ROOT}/${NAME}"

    echo "=========================================="
    echo "Validating checkpoint: ${CKPT_PATH}"
    echo "Result folder: ${RESULT_FOLDER}"
    echo "=========================================="

    for TEST_DATA in "${TEST_LIST[@]}"; do
        echo "------------------------------------------"
        echo "Testing on ${TEST_DATA}, p=${P}"
        echo "------------------------------------------"

        CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
            --p="${P}" \
            --test_data="${TEST_DATA}" \
            --arch="${ARCH}" \
            --ckpt="${CKPT_PATH}" \
            --result_folder="${RESULT_FOLDER}" \
            --select_k="${SELECT_K}" \
            --batch_size="${BATCH_SIZE}"
    done

    echo "Finished p=${P}"
    echo
done

echo "All experiments finished."