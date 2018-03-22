#!/usr/bin/env bash

mode="back"

seed=110
export CUDA_VISIBLE_DEVICES=0

batch_size=64
learning_rate=0.001
learning_rate_decay_rate=1.0

model_save_path=models/permuted_mnist_lstm_seed_"$seed"

log_path=log/permuted_mnist_lstm_seed_"$seed".txt

if [ "$mode" == "back" ]
then
    nohup python tf_permuted_mnist_lstm.py --seed $seed \
                                           --batch_size $batch_size \
                                           --learning_rate $learning_rate \
                                           --learning_rate_decay_rate $learning_rate_decay_rate \
                                           --model_save_path "$model_save_path" \
                                           > "$log_path" 2>&1 &
else
    python tf_permuted_mnist_lstm.py --seed $seed \
                                     --batch_size $batch_size \
                                     --learning_rate $learning_rate \
                                     --learning_rate_decay_rate $learning_rate_decay_rate \
                                     --model_save_path "$model_save_path"
fi
