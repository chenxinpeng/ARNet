#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

beam_file_path=data/splits/coco_test.txt
beam_model_path=models/ende_v4_seed_116/model_epoch-33.pth
beam_json_path=models/ende_inception_v4_seed_116/model_epoch-33_beam_offline_test.json

python3.6 image_caption_ende_infb.py --beam_file_path "$beam_file_path" \
                                     --beam_model_path "$beam_model_path" \
                                     --beam_json_path "$beam_json_path"
echo ${beam_file_path}
echo ${beam_model_path}
echo ${beam_json_path}
