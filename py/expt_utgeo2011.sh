#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES='2' python gcncrf_transductive.py --model crf \
  --learning_rate 0.01 --dataset utgeo2011 --deep --hidden1 600 --hidden2 600 \
  --output output/utgeo2011/pred_TCSDEEP_lr0.01_600_600.txt

