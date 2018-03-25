#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

vis_batch_size=1
vis_model_path=
vis_save_path=

python3.6 code_caption_ende_vis.py --vis_batch_size $vis_batch_size \
                                   --vis_model_path "$vis_model_path" \
                                   --vis_save_path "$vis_save_path"
