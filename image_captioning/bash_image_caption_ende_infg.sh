#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

idx=20
seed=110

infer_file_path=data/splits/coco_test.txt
infer_model_path=models/ende_inception_v4_seed_"$seed"/model_epoch-"$idx".pth
infer_json_path=models/ende_inception_v4_seed_"$seed"/model_epoch-"$idx"_greedy_offline_test.json

python3.6 image_caption_ende_infg.py --infer_file_path "$infer_file_path" \
                                     --infer_model_path "$infer_model_path" \
                                     --infer_json_path "$infer_json_path"
echo ${infer_file_path}
echo ${infer_model_path}
echo ${infer_json_path}
