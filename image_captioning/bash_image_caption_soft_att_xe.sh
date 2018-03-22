#!/usr/bin/env bash

mode="back0"
export CUDA_VISIBLE_DEVICES=0
seed=110
lstm_size=512
version=offline

xe_model_save_path=models/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"
train_json_path=models/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"/xe_train_early_stop.json

log_path=log/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_att.py --feature_type inception_v4 \
                                              --seed $seed \
                                              --version "$version" \
                                              --lstm_size $lstm_size \
                                              --xe_model_save_path "$xe_model_save_path" \
                                              --train_json_path "$train_json_path" \
                                              > "$log_path" 2>&1 &
else
    python3.6 image_caption_soft_att.py --feature_type inception_v4 \
                                        --seed $seed \
                                        --version "$version" \
                                        --lstm_size $lstm_size \
                                        --xe_model_save_path "$xe_model_save_path" \
                                        --train_json_path "$train_json_path"
fi
