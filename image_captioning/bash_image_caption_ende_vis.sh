#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

vis_batch_size=1
vis_model_path=models/encoder_decoder_inception_v4_seed_116/model_epoch-33.pth
vis_save_path=models/encoder_decoder_inception_v4_seed_116/model_epoch-33_hidden_states.pkl

python3.6 image_caption_ende_vis.py --vis_batch_size $vis_batch_size \
                                    --vis_model_path "$vis_model_path" \
                                    --vis_save_path "$vis_save_path"
