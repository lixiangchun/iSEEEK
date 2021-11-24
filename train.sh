#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.launch --nproc_per_node=2 --use_env train_mlm.py \
--train-file ../bert/gene_rank_2021Jun15.txt --val-file ../bert/gene_rank_test.txt -b 64 \
--lr 1.0e-4 --epochs 48 -j 2 --max-len 256 --num_warmup_steps 10000 \
--output-dir checkpoint --dict gene_rank_2021Jun15.json --print-freq 100

