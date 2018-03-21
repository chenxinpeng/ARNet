#!/usr/bin/env bash

mode="back"

seed=110
c_ratio=0.05
h_ratio=0.05
version=offline
export CUDA_VISIBLE_DEVICES=0

xe_model_save_path=models/soft_attention_zoneout_inception_v4_seed_"$seed"_zoneout_c_"$c_ratio"_h_"$h_ratio"
train_json_path=models/soft_attention_zoneout_inception_v4_seed_"$seed"_zoneout_c_"$c_ratio"_h_"$h_ratio"/xe_train_early_stop.json

log_path=log/soft_attention_inception_v4_seed_"$seed"_zoneout_c_"$c_ratio"_h_"$h_ratio".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_attention_zoneout.py --caption_model ShowAttendTellModel \
                                                            --function train_xe \
                                                            --feature_type inception_v4 \
                                                            --seed $seed \
                                                            --c_ratio $c_ratio \
                                                            --h_ratio $h_ratio \
                                                            --version "$version" \
                                                            --xe_model_save_path "$xe_model_save_path" \
                                                            > "$log_path" 2>&1 &
else
    python3.6 image_caption_soft_attention_zoneout.py --caption_model ShowAttendTellModel \
                                                      --function train_xe \
                                                      --feature_type inception_v4 \
                                                      --seed $seed \
                                                      --c_ratio $c_ratio \
                                                      --h_ratio $h_ratio \
                                                      --version "$version" \
                                                      --xe_model_save_path "$xe_model_save_path"
fi
