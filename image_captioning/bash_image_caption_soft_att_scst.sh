#!/usr/bin/env bash

# 110: 12
# 111: 11
# 112: 10
# 113: 12
# 114: 9
# 115: 12
# 116: 14
# 117: 8

mode="back"

idx=12
seed=110
version=offline
scst_batch_size=16

export CUDA_VISIBLE_DEVICES=0

scst_base_model_path=models/soft_attention_inception_v4_seed_"$seed"/model_epoch-"$idx".pth

scst_model_save_path=models/soft_attention_inception_v4_seed_"$seed"_scst

scst_train_json_path=models/soft_attention_inception_v4_seed_"$seed"_scst/scst_train_early_stop.json

log_path=log/soft_attention_inception_v4_seed_"$seed"_scst.txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_attention.py --caption_model ShowAttendTellModel \
                                                    --function train_scst \
                                                    --feature_type inception_v4 \
                                                    --seed $seed \
                                                    --version "$version" \
                                                    --scst_batch_size $scst_batch_size \
                                                    --scst_base_model_path "$scst_base_model_path" \
                                                    --scst_model_save_path "$scst_model_save_path" \
                                                    --scst_train_json_path "$scst_train_json_path" \
                                                    > "$log_path" 2>&1 &
else
    python3.6 image_caption_soft_attention.py --caption_model ShowAttendTellModel \
                                              --function train_scst \
                                              --feature_type inception_v4 \
                                              --seed $seed \
                                              --version "$version" \
                                              --scst_batch_size $scst_batch_size \
                                              --scst_base_model_path "$scst_base_model_path" \
                                              --scst_model_save_path "$scst_model_save_path" \
                                              --scst_train_json_path "$scst_train_json_path"
fi
