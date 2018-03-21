#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

t_SNE_batch_size=100

t_SNE_model_path=models/encoder_decoder_zoneout_inception_v4_seed_110_zoneout_c_0.05_h_0.05/model_epoch-9.pth
t_SNE_save_path=models/encoder_decoder_zoneout_inception_v4_seed_110_zoneout_c_0.05_h_0.05/model_epoch-9_hidden_states.pkl

python3.6 image_caption_soft_attention.py --function t_SNE_visilization \
                                          --t_SNE_batch_size $t_SNE_batch_size \
                                          --t_SNE_model_path "$t_SNE_model_path" \
                                          --t_SNE_save_path "$t_SNE_save_path"
