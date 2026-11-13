#!/bin/bash
# 实验二(ProGAN):复杂度/纹理控制,||δ|| 从零重算,不复用实验一结果
#   Stage 1  bpp(JPEG q75)+ Sobel
#   Stage 2  ||δ||(selector argmax 窗口,10 次共享重排)
#   Stage 3  KS / 分层 AUROC / 匹配 AUROC / 残差 AUROC
#   Stage 4  JPEG q95 统一重编码后重跑一遍

CUDA_ID="0"
TEST_DATA="ForenSynths"
DATASET_KEYS="progan"
CKPT_PATH="./checkpoints/ptrain_3090_0_1/best_model.pth"
RESULT_FOLDER="./results/complexity_control/progan"

MAX_SAMPLE=1000     # 每类数量,共 2000 张
NUM_REPS=10
BATCH_SIZE=128
P=0.1
SEED=0

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 analysis/complexity_control.py \
    --test_data="${TEST_DATA}" \
    --dataset_keys="${DATASET_KEYS}" \
    --ckpt="${CKPT_PATH}" \
    --arch="CLIP:ViT-L/14" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size="${BATCH_SIZE}" \
    --max_sample="${MAX_SAMPLE}" \
    --num_reps="${NUM_REPS}" \
    --num_bins=5 \
    --p="${P}" \
    --seed="${SEED}" \
    --format_check
