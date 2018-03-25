#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

test_model_path=models/encode_decode_seed_110/model_epoch-15.pth

python3.6 code_caption_ende_inf.py --test_model_path "$test_model_path"
