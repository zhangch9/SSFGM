#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES='2' python gcncrf_transductive.py --model crf \
  --learning_rate 0.01 --dataset world --deep --hidden1 1000 --hidden2 1000 \
  --output output/world/pred_TCSDEEP_lr0.01.txt
