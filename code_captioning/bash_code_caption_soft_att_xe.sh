#!/usr/bin/env bash

mode="back"
export CUDA_VISIBLE_DEVICES=0

seed=110
batch_size=16
learning_rate=0.001

model_save_path=models/soft_att_seed_"$seed"
log_path=log/soft_att_seed_"$seed".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 code_caption_soft_att_xe.py --seed $seed \
                                                --batch_size $batch_size \
                                                --learning_rate $learning_rate \
                                                --model_save_path "$model_save_path" \
                                                > "$log_path" 2>&1 &
else
    python3.6 code_caption_soft_att_xe.py --seed $seed \
                                          --batch_size $batch_size \
                                          --learning_rate $learning_rate \
                                          --model_save_path "$model_save_path"
fi
