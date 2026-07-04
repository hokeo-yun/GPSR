#!/bin/bash
# class_labels = ['tvmonitor', 'horse']
NAME="setting1_3090"
TOTAL_EPOCHS=5
TRAIN_CUDA=1
CHECKPOINTS_DIR="checkpoints/"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"
# DRCT: lr=0.0001, UFD: lr=0.00005
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
    --p=1 \
    --name="${NAME}" \
    --wang2020_data_path=${DATA_DIR} \
    --checkpoints_dir="${CHECKPOINTS_DIR}" \
    --data_mode="sd1_4" \
    --arch="CLIP:ViT-L/14" \
    --lr=0.00005 \
    --fix_backbone \
    --select_k=5 \
    --batch_size=256 \
    --save_epoch_freq=5 \
    --niter="${TOTAL_EPOCHS}"

CUDA_ID="1"
CKPT_PATH="./checkpoints/setting1_3090/best_model.pth"
RESULT_FOLDER="./results/setting1_3090/"

TEST_DATA="UFD"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=1 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256