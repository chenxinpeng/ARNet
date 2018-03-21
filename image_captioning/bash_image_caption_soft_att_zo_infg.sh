#!/usr/bin/env bash

# 110: 8
# 111: 8
# 112: 7
# 113: 13
# 114: 9
# 115: 7
# 116: 8
# 117: 11

idx=8
seed=110
c_ratio=0.05
h_ratio=0.05
export CUDA_VISIBLE_DEVICES=0

infer_file_path=data/splits/coco_test.txt
infer_model_path=models/soft_attention_zoneout_inception_v4_seed_"$seed"_zoneout_c_"$c_ratio"_h_"$h_ratio"/model_epoch-"$idx".pth
infer_json_path=models/soft_attention_zoneout_inception_v4_seed_"$seed"_zoneout_c_"$c_ratio"_h_"$h_ratio"/model_epoch-"$idx"_greedy_offline_test.json

python3.6 image_caption_soft_attention_zoneout.py --caption_model ShowAttendTellModel \
                                                  --function inference \
                                                  --infer_file_path "$infer_file_path" \
                                                  --infer_model_path "$infer_model_path" \
                                                  --infer_json_path "$infer_json_path"
echo $infer_file_path
echo $infer_model_path
echo $infer_json_path
