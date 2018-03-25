#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

test_model_path=models/soft_att_seed_110/model_epoch-38.pth

python3.6 code_caption_soft_att_inf.py --batch_size 1 --test_model_path "$test_model_path"

echo ${test_model_path}
