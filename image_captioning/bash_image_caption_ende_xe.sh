#!/usr/bin/env bash

mode="back"
export CUDA_VISIBLE_DEVICES=0

seed=110
version=offline

xe_model_save_path=models/ende_inception_v4_seed_"$seed"
train_json_path=models/ende_inception_v4_seed_"$seed"/xe_train_early_stop.json

log_path=log/ende_inception_v4_seed_"$seed"

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_ende_xe.py --feature_type inception_v4 \
                                             --seed $seed \
                                             --version "$version" \
                                             --xe_model_save_path "$xe_model_save_path" \
                                             > "$log_path" 2>&1 &
else
    python3.6 image_caption_ende_xe.py --feature_type inception_v4 \
                                       --seed $seed \
                                       --version "$version" \
                                       --xe_model_save_path "$xe_model_save_path"
fi
