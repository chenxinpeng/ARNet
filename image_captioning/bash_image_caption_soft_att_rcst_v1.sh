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

reconstruct_weight=0.005
reconstruct_learning_rate=0.0001

export CUDA_VISIBLE_DEVICES=0

reconstruct_model_path=models/soft_attention_inception_v4_seed_"$seed"/model_epoch-"$idx"_reconstruct_lstm_1st.pth
reconstruct_model_save_path=models/soft_attention_inception_v4_seed_"$seed"_reconstruct_"$reconstruct_weight"
reconstruct_train_json_path=models/soft_attention_inception_v4_seed_"$seed"_reconstruct_"$reconstruct_weight"/reconstruct_train_early_stop.json

log_path=log/soft_attention_inception_v4_seed_"$seed"_reconstruct_"$reconstruct_weight".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_attention_reconstruct_lstm_1st.py --caption_model ShowAttendTellModel \
                                                                         --function reconstruct \
                                                                         --feature_type inception_v4 \
                                                                         --seed $seed \
                                                                         --version "$version" \
                                                                         --reconstruct_weight $reconstruct_weight \
                                                                         --reconstruct_learning_rate $reconstruct_learning_rate \
                                                                         --reconstruct_model_path "$reconstruct_model_path" \
                                                                         --reconstruct_model_save_path "$reconstruct_model_save_path" \
                                                                         --reconstruct_train_json_path "$reconstruct_train_json_path" \
                                                                         > "$log_path" 2>&1 &
else
    python3.6 image_caption_soft_attention_reconstruct_lstm_1st.py --caption_model ShowAttendTellModel \
                                                                   --function reconstruct \
                                                                   --feature_type inception_v4 \
                                                                   --seed $seed \
                                                                   --version "$version" \
                                                                   --reconstruct_weight $reconstruct_weight \
                                                                   --reconstruct_learning_rate $reconstruct_learning_rate \
                                                                   --reconstruct_model_path "$reconstruct_model_path" \
                                                                   --reconstruct_model_save_path "$reconstruct_model_save_path" \
                                                                   --reconstruct_train_json_path "$reconstruct_train_json_path"
fi
