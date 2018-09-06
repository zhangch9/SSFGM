#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES='2' python gcncrf_transductive.py --model crf \
  --learning_rate 0.01 --dataset twitter_usa \
  --output output/twitter_usa/pred_TCS_lr0.01.txt
#CUDA_VISIBLE_DEVICES='2' python gcncrf_transductive.py --model gcn_crf \
#  --learning_rate 0.01 --dataset twitter_usa \
#  --output output/twitter_usa/pred_GCNCRF_lr0.01.txt

