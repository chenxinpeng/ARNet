#!/usr/bin/env bash

mode="back"

export CUDA_VISIBLE_DEVICES=7
idx=8
seed=117
lstm_size=512
batch_size=400
version=offline

rcst_time=3
rcst_lr=0.0001
rcst_scalar=0.0001
lr_decay_every=3
lr_decay_rate=1.0

rcst_model_save_path=models/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"_rcst_3rd_"$rcst_scalar"_rcst_time_"$rcst_time"_scratch
rcst_train_json_path=models/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"_rcst_3rd_"$rcst_scalar"_rcst_time_"$rcst_time"_scratch/rcst_train_early_stop.json

log_path=log/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"_rcst_3rd_"$rcst_scalar"_rcst_time_"$rcst_time"_scratch.txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_att_rcst.py --feature_type inception_v4 \
                                                   --seed ${seed} \
                                                   --lstm_size ${lstm_size} \
                                                   --batch_size ${batch_size} \
                                                   --version "$version" \
                                                   --rcst_time ${rcst_time} \
                                                   --reconstruct_weight ${rcst_scalar} \
                                                   --reconstruct_learning_rate ${rcst_lr} \
                                                   --learning_rate_decay_every ${lr_decay_every} \
                                                   --learning_rate_decay_rate ${lr_decay_rate} \
                                                   --reconstruct_model_save_path "$rcst_model_save_path" \
                                                   --reconstruct_train_json_path "$rcst_train_json_path" \
                                                   > "$log_path" 2>&1 &
else
    python3.6 image_caption_soft_att_rcst.py --feature_type inception_v4 \
                                             --seed ${seed} \
                                             --lstm_size ${lstm_size} \
                                             --batch_size ${batch_size} \
                                             --version "$version" \
                                             --rcst_time ${rcst_time} \
                                             --reconstruct_weight ${rcst_scalar} \
                                             --reconstruct_learning_rate ${rcst_lr} \
                                             --learning_rate_decay_every ${lr_decay_every} \
                                             --learning_rate_decay_rate ${lr_decay_rate} \
                                             --reconstruct_model_save_path "$rcst_model_save_path" \
                                             --reconstruct_train_json_path "$rcst_train_json_path"
fi
