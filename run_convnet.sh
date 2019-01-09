#!/bin/sh

python -u train.py \
    --model Convnet \
    --kernel-size 1 \
    --hidden-dim 10 \
    --epochs 10 \
    --weight-decay 0.0 \
    --momentum 0.0 \
    --batch-size 512 \
    --lr 0.01 | tee convnet.log