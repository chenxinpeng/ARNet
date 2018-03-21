#!/usr/bin/env bash

mode="back"

idx=6
seed=110
lstm_size=512
version=offline

rcst_lr=0.0001
rcst_scalar=0.005
lr_decay_every=3
lr_decay_rate=1.0

export CUDA_VISIBLE_DEVICES=0

rcst_model_path=models/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"/model_epoch-"$idx".pth
rcst_model_save_path=models/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"_rcst_3rd_"$rcst_scalar"
rcst_train_json_path=models/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"_rcst_3rd_"$rcst_scalar"/rcst_train_early_stop.json

log_path=log/soft_attention_inception_v4_seed_"$seed"_h_"$lstm_size"_rcst_3rd_"$rcst_scalar".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_att_rcst.py --feature_type inception_v4 \
                                                   --seed ${seed} \
                                                   --lstm_size ${lstm_size} \
                                                   --version "$version" \
                                                   --reconstruct_weight ${rcst_scalar} \
                                                   --reconstruct_learning_rate ${rcst_lr} \
                                                   --learning_rate_decay_every ${lr_decay_every} \
                                                   --learning_rate_decay_rate ${lr_decay_rate} \
                                                   --reconstruct_model_path "$rcst_model_path" \
                                                   --reconstruct_model_save_path "$rcst_model_save_path" \
                                                   --reconstruct_train_json_path "$rcst_train_json_path" \
                                                   > "$log_path" 2>&1 &
else
    python3.6 image_caption_soft_att_rcst.py --feature_type inception_v4 \
                                             --seed ${seed} \
                                             --lstm_size ${lstm_size} \
                                             --version "$version" \
                                             --reconstruct_weight ${rcst_scalar} \
                                             --reconstruct_learning_rate ${rcst_lr} \
                                             --learning_rate_decay_every ${lr_decay_every} \
                                             --learning_rate_decay_rate ${lr_decay_rate} \
                                             --reconstruct_model_path "$rcst_model_path" \
                                             --reconstruct_model_save_path "$rcst_model_save_path" \
                                             --reconstruct_train_json_path "$rcst_train_json_path"
fi
