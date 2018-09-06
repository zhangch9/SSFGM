#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES='5' python gcncrf_transductive.py --model crf \
#  --learning_rate 0.01 --dataset geotext --deep --hidden1 300 --hidden2 300 \
#  --output output/geotext/pred_TCSDEEP_lr0.01_300_300.txt
CUDA_VISIBLE_DEVICES='5' python gcncrf_transductive.py --model gcn_crf \
  --learning_rate 0.01 --dataset geotext \
  --output output/geotext/pred_GCNCRF_lr0.01.txt
