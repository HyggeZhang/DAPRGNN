#!/bin/sh

GPU=0

echo "=====Cora====="
CUDA_VISIBLE_DEVICES=${GPU} python daprgnn.py --dataset=Cora --weight_decay=0.22 --K=10 --dropout=0.85 --lr=0.005

echo "=====CiteSeer====="
CUDA_VISIBLE_DEVICES=${GPU} python daprgnn.py --dataset=CiteSeer --weight_decay=0.015 --K=10 --dropout=0.5 --lr=0.005

echo "=====PubMed====="
CUDA_VISIBLE_DEVICES=${GPU} python daprgnn.py --dataset=PubMed --weight_decay=0.028 --K=20 --dropout=0.83 --lr=0.02

echo "=====Wisconsin====="
CUDA_VISIBLE_DEVICES=${GPU} python daprgnn.py --dataset=wisconsin --weight_decay=0.002 --K=10 --dropout=0.34 --lr=0.016

echo "=====Texas====="
CUDA_VISIBLE_DEVICES=${GPU} python daprgnn.py --dataset=texas --weight_decay=0.002 --K=10 --dropout=0.34 --lr=0.016

echo "=====Cornell====="
CUDA_VISIBLE_DEVICES=${GPU} python daprgnn.py --dataset=cornell --weight_decay=0.003 --K=10 --dropout=0.6 --lr=0.008

echo "=====Actor====="
CUDA_VISIBLE_DEVICES=${GPU} python daprgnn.py --dataset=actor --weight_decay=0.002 --K=10 --dropout=0.45 --lr=0.016







