#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

vis_batch_size=80
vis_model_path=models/soft_attention_inception_v4_seed_117/model_epoch-8.pth
vis_save_path=models/soft_attention_inception_v4_seed_117/model_epoch-8_hidden_states.pkl

python3.6 image_caption_soft_att_vis.py --vis_batch_size $vis_batch_size \
                                        --vis_model_path "$vis_model_path" \
                                        --vis_save_path "$vis_save_path"
