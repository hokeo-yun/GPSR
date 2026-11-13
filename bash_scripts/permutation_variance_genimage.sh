#!/bin/bash
# permutation_variance:GenImage(Stable Diffusion v1.4 子集)
#   across samples     :固定 10 次随机,每样本先平均,再统计 N 个样本的均值/方差
#   across permutation :固定全部样本,每次扰动先平均,再统计 10 次的均值/方差

CUDA_ID="1"
TEST_DATA="GenImage"
DATASET_KEYS="stable_diffusion_v_1_4"
CKPT_PATH="./checkpoints/ptrain_3090_0_1/best_model.pth"
RESULT_FOLDER="./results/permutation_variance/sdv1_4"

NUM_REPS=10
# 注意:max_sample 是「每个类别」的数量,2000 张 = 1000 real + 1000 fake
# 快速验证时可设 100(共 200 张)
MAX_SAMPLE=1000
BATCH_SIZE=128
P=0.1
SEED=0

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 analysis/permutation_variance.py \
    --test_data="${TEST_DATA}" \
    --dataset_keys="${DATASET_KEYS}" \
    --ckpt="${CKPT_PATH}" \
    --arch="CLIP:ViT-L/14" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size="${BATCH_SIZE}" \
    --max_sample="${MAX_SAMPLE}" \
    --num_reps="${NUM_REPS}" \
    --mode="ps" \
    --perm_mode="shared" \
    --p="${P}" \
    --seed="${SEED}" \
    --plot
