#!/usr/bin/env bash

mode="back"
export CUDA_VISIBLE_DEVICES=0

idx=39
seed=110

batch_size=16
rcst_weight=0.005
learning_rate=0.0005

rcst_model_base_path=models/soft_att_seed_"$seed"/model_epoch-"$idx".pth
rcst_model_save_path=models/soft_att_seed_"$seed"_rcst_"$rcst_weight"
log_path=log/soft_att_seed_"$seed"_rcst_"$rcst_weight".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 code_caption_soft_att_rcst.py --seed $seed \
                                                  --batch_size $batch_size \
                                                  --learning_rate $learning_rate \
                                                  --rcst_weight $rcst_weight \
                                                  --rcst_model_base_path "$rcst_model_base_path" \
                                                  --rcst_model_save_path "$rcst_model_save_path" \
                                                  > "$log_path" 2>&1 &
else
    python3.6 code_caption_soft_att_rcst.py --seed $seed \
                                            --batch_size $batch_size \
                                            --learning_rate $learning_rate \
                                            --rcst_weight $rcst_weight \
                                            --rcst_model_base_path "$rcst_model_base_path" \
                                            --rcst_model_save_path "$rcst_model_save_path"
fi
