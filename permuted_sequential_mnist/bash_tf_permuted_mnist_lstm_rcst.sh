#!/usr/bin/env bash

mode="back"
export CUDA_VISIBLE_DEVICES=0

idx=61
seed=110
batch_size=64
learning_rate=0.001
learning_rate_decay_rate=1.0

rcst_weight=0.001
rcst_model_base_path=models/permuted_mnist_lstm_seed_"$seed"/model_epoch-"$idx"_rcst
rcst_model_save_path=models/permuted_mnist_lstm_seed_"$seed"_rcst_"$reconstruct_weight"

log_path=log/seq_mnist_lstm_seed_"$seed"_rcst_"$rcst_weight".txt

if [ "$mode" == "back" ]
then
    nohup python tf_permuted_mnist_lstm_rcst.py --seed $seed \
                                                  --batch_size $batch_size \
                                                  --learning_rate $learning_rate \
                                                  --learning_rate_decay_rate $learning_rate_decay_rate \
                                                  --rcst_weight $rcst_weight \
                                                  --rcst_model_base_path "$rcst_model_base_path" \
                                                  --rcst_model_save_path "$rcst_model_save_path" \
                                                  > "$log_path" 2>&1 &
else
    python tf_permuted_mnist_lstm_rcst.py --seed $seed \
                                          --batch_size $batch_size \
                                          --learning_rate $learning_rate \
                                          --learning_rate_decay_rate $learning_rate_decay_rate \
                                          --rcst_weight $rcst_weight \
                                          --rcst_model_base_path "$rcst_model_base_path" \
                                          --rcst_model_save_path "$rcst_model_save_path"
fi
