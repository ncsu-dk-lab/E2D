#!/bin/bash

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ARCH_PATH="./model/imagenet-21k_resnet18.pth" # Path to ImageNet-21K pre-trained model
SYN_DATA_PATH="..." # Path to save synthesized data
INIT_DIR=".../imagenet21k_resized/imagenet21k_train" # Path to ImageNet-21K training data for initialization
EXP_NAME="E2D_IPC10_RN18_Imagenet21K" # Experiment name

SAVE_OUTPUT_DIR="..." # Path to save trained models and outputs
SYN_DATA_DIR="../$SYN_DATA_PATH$EXP_NAME" # Path to synthesized data directory
VAL_DIR=".../imagenet21k_resized/imagenet21k_val" # Path to ImageNet-21K validation data


# ------------------------------------------------------------------
#  Synthesis (Recovery)
# ------------------------------------------------------------------

python recover.py \
    --arch-name "resnet18" \
    --arch-path "$ARCH_PATH" \
    --exp-name "$EXP_NAME" \
    --syn-data-path "$SYN_DATA_PATH" \
    --batch-size 80 \
    --lr 0.005 \
    --r-bn 0.01 \
    --iteration 400 \
    --K 280 \
    --loss-threshold 0.3 \
    --store-best-images \
    --initial-img-dir "$INIT_DIR" \
    --ipc-start 0 \
    --ipc-end 10 \
    --G 0

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

wandb enabled
wandb offline

cd validate

python train_KD.py \
    --wandb-project "val_rn18_kd" \
    --batch-size 100 \
    --gradient-accumulation-steps 1 \
    --model resnet18 \
    --teacher-model resnet18 \
    -j 2 \
    -T 20 \
    --st 2 \
    --ls-type cos2 \
    --IPC 10 \
    --adamw-lr 0.002 \
    --epochs 300 \
    --mix-type cutmix \
    --output-dir "$SAVE_OUTPUT_DIR" \
    --train-dir "$SYN_DATA_DIR" \
    --arch-path "$ARCH_PATH" \
    --val-dir "$VAL_DIR"

