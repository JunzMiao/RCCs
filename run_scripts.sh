#!/usr/bin/env bash

TRAIN_ARGS="
    --pretrained_model_path="pretrained_model.pth" \
    --model_name="ResNet_CIFAR10" \
    --dataset_name="cifar10" \
    --dataset_path="dataset" \
    --class_num=10 \
    --optimizer="Adam" \
    --learning_rate=0.001 \
    --momentum=0.9 \
    --weight_decay=5e-4 \
    --lr_scheduler=1 \
    --lr_step_size=30 \
    --gamma=0.1 \
    --max_iter=100 \
    --train_batch_size=128 \
    --test_batch_size=128 \
    --device="cuda:0" \
    --output_dir="trained_model.pth" \
"

ADV_ARGS="
    --attacker="bim" \
    --norm=0 \
    --eps=0.05 \
    --loss="ce" \
    --step_size="0.01" \
    --steps=10 \
    --decay_factor=0.9 \
    --resize_rate=0.85 \
    --coeff_adv_loss=0.5 \
    --dis_metric=euc \
    --coeff_local_constraint=15 \
"

python train.py $TRAIN_ARGS $ADV_ARGS
