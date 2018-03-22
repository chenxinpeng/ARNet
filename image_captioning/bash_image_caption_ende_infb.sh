#!/usr/bin/env bash

# 110: 20
# 111: 32
# 112: 23
# 113: 34
# 114: 34
# 115: 27
# 116: 33
# 117: 25

idx=20
seed=110
export CUDA_VISIBLE_DEVICES=0

beam_file_path=data/splits/coco_test.txt
beam_model_path=models/encoder_decoder_inception_v4_seed_"$seed"/model_epoch-"$idx".pth
beam_json_path=models/encoder_decoder_inception_v4_seed_"$seed"/model_epoch-"$idx"_beam_offline_test.json

python3.6 image_caption_encoder_decoder.py --caption_model encoder_decoder \
                                           --function beam_search \
                                           --beam_file_path "$beam_file_path" \
                                           --beam_model_path "$beam_model_path" \
                                           --beam_json_path "$beam_json_path"
echo ${beam_file_path}
echo ${beam_model_path}
echo ${beam_json_path}
