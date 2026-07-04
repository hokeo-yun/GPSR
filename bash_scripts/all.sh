NAME="p1"
TOTAL_EPOCHS=5
TRAIN_CUDA=0
CHECKPOINTS_DIR="checkpoints/p"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"
# DRCT: lr=0.0001, UFD: lr=0.00005
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
    --p=0.2 \
    --lambdav=0 \
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

CUDA_ID="0"
CKPT_PATH="./checkpoints/p/p1/best_model.pth"
RESULT_FOLDER="./results/p/p1/"

TEST_DATA="GenImage"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=0.2 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256

NAME="p2"
TOTAL_EPOCHS=5
TRAIN_CUDA=0
CHECKPOINTS_DIR="checkpoints/p"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"
# DRCT: lr=0.0001, UFD: lr=0.00005
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
    --p=0.4 \
    --lambdav=0 \
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

CUDA_ID="0"
CKPT_PATH="./checkpoints/p/p2/best_model.pth"
RESULT_FOLDER="./results/p/p2/"

TEST_DATA="GenImage"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=0.4 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256

NAME="p3"
TOTAL_EPOCHS=5
TRAIN_CUDA=0
CHECKPOINTS_DIR="checkpoints/p"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"
# DRCT: lr=0.0001, UFD: lr=0.00005
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
    --p=0.6 \
    --lambdav=0 \
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

CUDA_ID="0"
CKPT_PATH="./checkpoints/p/p3/best_model.pth"
RESULT_FOLDER="./results/p/p3/"

TEST_DATA="GenImage"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=0.6 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256

NAME="p4"
TOTAL_EPOCHS=5
TRAIN_CUDA=0
CHECKPOINTS_DIR="checkpoints/p"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"
# DRCT: lr=0.0001, UFD: lr=0.00005
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
    --p=0.8 \
    --lambdav=0 \
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

CUDA_ID="0"
CKPT_PATH="./checkpoints/p/p4/best_model.pth"
RESULT_FOLDER="./results/p/p4/"

TEST_DATA="GenImage"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" python3 validate.py \
    --p=0.8 \
    --test_data="${TEST_DATA}" \
    --arch="CLIP:ViT-L/14" \
    --ckpt="${CKPT_PATH}" \
    --result_folder="${RESULT_FOLDER}" \
    --select_k=5 \
    --batch_size=256

NAME="p6"
TOTAL_EPOCHS=5
TRAIN_CUDA=0
CHECKPOINTS_DIR="checkpoints/p"
DATA_DIR="/home/108/u108009/dataset/ForenSynths"
# DRCT: lr=0.0001, UFD: lr=0.00005
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA}" python train.py \
    --p=1 \
    --lambdav=0 \
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