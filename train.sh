#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train-iseeek.py \
--train-file train.txt --val-file test.txt -b 64 \
--wd 0.01 --lr 1.0e-4 --epochs 48 -j 2 --max-len 128 --num_warmup_steps 10000 \
--output-dir checkpoint --print-freq 100

