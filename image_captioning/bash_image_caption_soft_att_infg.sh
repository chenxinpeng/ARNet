#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=7
idx=8
seed=117
lstm_size=512

infer_file_path=data/splits/coco_test.txt
infer_model_path=models/soft_attention_inception_v4_seed_"$seed"/model_epoch-"$idx".pth
infer_json_path=models/soft_attention_inception_v4_seed_"$seed"/model_epoch-"$idx"_greedy_offline_test.json

python3.6 image_caption_soft_att_infg.py --function inference \
                                         --lstm_size $lstm_size \
                                         --infer_file_path "$infer_file_path" \
                                         --infer_model_path "$infer_model_path" \
                                         --infer_json_path "$infer_json_path"
echo $infer_file_path
echo $infer_model_path
echo $infer_json_path
