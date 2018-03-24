#!/usr/bin/env bash

mode="back"
export CUDA_VISIBLE_DEVICES=0

seed=110
learning_rate=0.001

model_save_path=models/ende_seed_"$seed"
log_path=log/ende_seed_"$seed".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 code_caption_ende_xe.py --seed $seed \
                                            --learning_rate $learning_rate \
                                            --model_save_path "$model_save_path" \
                                            > "$log_path" 2>&1 &
else
    python3.6 code_caption_ende_xe.py --seed $seed \
                                      --learning_rate $learning_rate \
                                      --model_save_path "$model_save_path"
fi
