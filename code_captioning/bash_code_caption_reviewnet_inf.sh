#!/usr/bin/env bash

idx=71
seed=110
export CUDA_VISIBLE_DEVICES=0

test_model_path=models/reviewnet_seed_"$seed"/model_epoch-"$idx".pth

python3.6 code_caption_reviewnet_inf.py --test_model_path "$test_model_path"

echo ${test_model_path}
