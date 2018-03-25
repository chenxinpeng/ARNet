#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

vis_batch_size=16
vis_model_path=models/soft_attention_seed_111/model_epoch-26.pth
vis_save_path=models/ssoft_attention_seed_111/model_epoch-26_hidden_states.pkl

python3.6 code_caption_soft_att_vis.py --vis_batch_size $vis_batch_size \
                                         --vis_model_path "$vis_model_path" \
                                         --vis_save_path "$vis_save_path"
