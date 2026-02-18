#!/bin/bash

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

EXP_NAME="E2D_IPC10_resnet18_ImageNet1K" # Experiment name
INIT_DIR=".../ImageNet-1K/train" # Path to ImageNet-1K training data for initialization
VAL_PATH=".../ImageNet-1K/val" # Path to ImageNet-1K validation data
SYN_PATH=".../syn_data" # Path to save synthesized data
RELABEL_PATH=".../relabel/FKD_${EXP_NAME}" # Path to save soft labels
SAVE_PATH=".../save" # Path to save trained models and outputs
# if not specified, deafult = ./statistic
STATISTIC_PATH="" # Path to dataset statistics



# ---------------------------------------------------------
# 1. Recovery
# ---------------------------------------------------------

cd recover

python recover.py \
    --arch-name "resnet18" \
    --exp-name "$EXP_NAME" \
    --batch-size 80 \
    --lr 0.05 \
    --category-aware "global" \
    --ipc-number 10 \
    --training-momentum 0.8 \
    --iteration 200 \
    --drop-rate 0.0 \
    --train-data-path "$INIT_DIR" \
    --l2-scale 0 --tv-l2 0 --r-loss 0.1 --nuc-norm 1.0 \
    --verifier \
    --AMP 1 \
    --store-best-images \
    --gpu-id 0,1,2,3 \
    --K 140 \
    --loss-threshold 0.5 \
    --syn-data-path "$SYN_PATH" \
    --initial-img-dir "$INIT_DIR" \
    --statistic-path "$STATISTIC_PATH"

# ---------------------------------------------------------
# 2. Relabel
# ---------------------------------------------------------

cd ../relabel

python generate_soft_label_with_db.py \
    -b 100 -j 8 --epochs 300 \
    --fkd-seed 42 \
    --input-size 224 \
    --min-scale-crops 0.5 --max-scale-crops 1 \
    --use-fp16 \
    --candidate-number 4 \
    --fkd-path "$RELABEL_PATH" \
    --mode fkd_save \
    --mix-type cutmix \
    --data "$SYN_PATH/$EXP_NAME"

# ---------------------------------------------------------
# 3. Training
# ---------------------------------------------------------

cd ../train
wandb offline

python train_FKD_parallel.py \
    --wandb-project "final_rn18_fkd_${EXP_NAME}" \
    --batch-size 100 \
    --model resnet18 \
    --ls-type cos2 \
    --loss-type mse_gt \
    --ce-weight 0.025 \
    -j 4 \
    --gradient-accumulation-steps 1 \
    --st 2 \
    --ema-dr 0.99 \
    -T 20 \
    --gpu-id 0 \
    --mix-type cutmix \
    --output-dir "$SAVE_PATH/fkd_${EXP_NAME}/" \
    --train-dir "$SYN_PATH/$EXP_NAME" \
    --val-dir "$VAL_PATH" \
    --fkd-path "$RELABEL_PATH"
