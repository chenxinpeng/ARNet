#!/usr/bin/env bash

idx=20
seed=110
t_SNE_batch_size=1
export CUDA_VISIBLE_DEVICES=0

t_SNE_model_path=models/encoder_decoder_inception_v4_seed_"$seed"/model_epoch-"$idx".pth
t_SNE_save_path=models/encoder_decoder_inception_v4_seed_"$seed"/model_epoch-"$idx"_hidden_states.pkl

python3.6 image_caption_encoder_decoder.py --function t_SNE_visilization \
                                           --t_SNE_batch_size $t_SNE_batch_size \
                                           --t_SNE_model_path "$t_SNE_model_path" \
                                           --t_SNE_save_path "$t_SNE_save_path"
