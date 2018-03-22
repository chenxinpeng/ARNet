#!/usr/bin/env bash

mode="back"
export CUDA_VISIBLE_DEVICES=0

idx=20
seed=110
version=offline

rcst_weight=0.005
rcst_learning_rate=0.0005

rcst_model_path=models/ende_v4_seed_"$seed"/model_epoch-"$idx".pth
rcst_model_save_path=models/ende_v4_seed_"$seed"_rcst_"$rcst_weight"
rcst_train_json_path=models/ende_v4_seed_"$seed"_rcst_"$rcst_weight"/rcst_train_early_stop.json

log_path=log/ende_v4_seed_"$seed"_rcst_"$rcst_weight".txt

if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_ende_rcst_lstm.py --feature_type inception_v4 \
                                                    --seed $seed \
                                                    --version "$version" \
                                                    --rcst_weight $rcst_weight \
                                                    --rcst_learning_rate $rcst_learning_rate \
                                                    --rcst_model_path "$rcst_model_path" \
                                                    --rcst_model_save_path "$rcst_model_save_path" \
                                                    --rcst_train_json_path "$rcst_train_json_path" \
                                                    > "$log_path" 2>&1 &
else
    python3.6 image_caption_ende_rcst_lstm.py --feature_type inception_v4 \
                                              --seed $seed \
                                              --version "$version" \
                                              --rcst_weight $rcst_weight \
                                              --rcst_learning_rate $rcst_learning_rate \
                                              --rcst_model_path "$rcst_model_path" \
                                              --rcst_model_save_path "$rcst_model_save_path" \
                                              --rcst_train_json_path "$rcst_train_json_path"
fi
