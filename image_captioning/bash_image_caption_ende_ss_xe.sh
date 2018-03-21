#!/usr/bin/env bash

mode="back"

seed=110
export CUDA_VISIBLE_DEVICES=0

version=offline
batch_size=160

xe_model_save_path=models/encoder_decoder_ss_inception_v4_seed_"$seed"
train_json_path=models/encoder_decoder_ss_inception_v4_seed_"$seed"/xe_train_early_stop.json

log_path=log/encoder_decoder_inception_ss_inception_v4_seed_"$seed".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_encoder_decoder_ss.py --caption_model encoder_decoder \
                                                        --function train_xe \
                                                        --feature_type inception_v4 \
                                                        --seed $seed \
                                                        --batch_size $batch_size \
                                                        --version "$version" \
                                                        --xe_model_save_path "$xe_model_save_path" \
                                                        > "$log_path" 2>&1 &
else
    python3.6 image_caption_encoder_decoder_ss.py --caption_model encoder_decoder \
                                                  --function train_xe \
                                                  --feature_type inception_v4 \
                                                  --seed $seed \
                                                  --batch_size $batch_size \
                                                  --version "$version" \
                                                  --xe_model_save_path "$xe_model_save_path"
fi
