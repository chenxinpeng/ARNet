#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

beam_file_path=data/splits/coco_test.txt
beam_model_path=models/soft_attention_inception_v4_seed_117/model_epoch-8.pth
beam_json_path=models/soft_attention_inception_v4_seed_117/model_epoch-8_beam_offline_test.json

python3.6 image_caption_soft_att_infb.py --lstm_size 512 \
                                         --beam_file_path "$beam_file_path" \
                                         --beam_model_path "$beam_model_path" \
                                         --beam_json_path "$beam_json_path"
echo $beam_file_path
echo $beam_model_path
echo $beam_json_path
