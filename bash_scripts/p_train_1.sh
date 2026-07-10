#!/bin/bash

set -e

# ================= 基础配置 =================
TOTAL_EPOCHS=5
TRAIN_CUDA=0
CUDA_ID=0

CHECKPOINTS_DIR="checkpoints"
RESULT_ROOT="results"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"

ARCH="CLIP:ViT-L/14"
LR=0.00005
SELECT_K=5
BATCH_SIZE=256
DATA_MODE="sd1_4"

# p 从 0.1 到 1.0
P_LIST=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# 测试集
TEST_LIST=("ForenSynths" "GenImage" "UFD")

# ================= 开始循环 =================
for P in "${P_LIST[@]}"; do

    # 把 0.1 转成 0_1，用于文件夹命名
    P_TAG=$(echo "${P}" | sed 's/\./_/g')

    NAME="ptrain_3090_${P_TAG}"

    echo "=========================================="
    echo "Start training: p=${P}"
    echo "Experiment name: ${NAME}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
        --p="${P}" \
        --name="${NAME}" \
        --wang2020_data_path="${DATA_DIR}" \
        --checkpoints_dir="${CHECKPOINTS_DIR}" \
        --data_mode="${DATA_MODE}" \
        --arch="${ARCH}" \
        --lr="${LR}" \
        --fix_backbone \
        --select_k="${SELECT_K}" \
        --batch_size="${BATCH_SIZE}" \
        --save_epoch_freq=5 \
        --niter="${TOTAL_EPOCHS}"

    CKPT_PATH="./${CHECKPOINTS_DIR}/${NAME}/best_model.pth"
    RESULT_FOLDER="./${RESULT_ROOT}/ptrain_3090/${NAME}"

    echo "=========================================="
    echo "Start validation for p=${P}"
    echo "Checkpoint: ${CKPT_PATH}"
    echo "Result folder: ${RESULT_FOLDER}"
    echo "=========================================="

    for TEST_DATA in "${TEST_LIST[@]}"; do
        echo "------------------------------------------"
        echo "Testing ${TEST_DATA} with p=${P}"
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

echo "All p-sweep experiments finished."