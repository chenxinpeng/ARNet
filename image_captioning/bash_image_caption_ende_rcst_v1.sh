#!/usr/bin/env bash

# 110: 20
# 111: 32
# 112: 23
# 113: 34
# 114: 34
# 115: 27
# 116: 33
# 117: 25

mode="back"

idx=20
seed=110
version=offline

reconstruct_weight=0.005
reconstruct_learning_rate=0.0001

export CUDA_VISIBLE_DEVICES=0

reconstruct_model_path=models/encoder_decoder_inception_v4_seed_"$seed"/model_epoch-"$idx"_reconstruct_lstm_1st.pth
reconstruct_model_save_path=models/encoder_decoder_inception_v4_seed_"$seed"_reconstruct_"$reconstruct_weight"
reconstruct_train_json_path=models/encoder_decoder_inception_v4_seed_"$seed"_reconstruct_"$reconstruct_weight"/reconstruct_train_early_stop.json

log_path=log/encoder_decoder_inception_v4_seed_"$seed"_reconstruct_"$reconstruct_weight".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_encoder_decoder_reconstruct_lstm_1st.py --caption_model encoder_decoder \
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
    python3.6 image_caption_encoder_decoder_reconstruct_lstm_1st.py --caption_model encoder_decoder \
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
