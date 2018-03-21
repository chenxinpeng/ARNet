#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
lstm_size=512

beam_file_path=data/splits/coco_test.txt
beam_json_path=log/ensemble_beam_offline_test.json

ensemble_model_0=models/soft_attention_inception_v4_seed_117_reconstruct_2nd_0.005/model_epoch-1_xe_model.pth
ensemble_model_1=models/soft_attention_inception_v4_seed_117_reconstruct_0.01/model_epoch-7_xe_model.pth
ensemble_model_2=models/soft_attention_inception_v4_seed_112_reconstruct_2nd_0.05/model_epoch-3_xe_model.pth
ensemble_model_3=models/soft_attention_inception_v4_seed_113_reconstruct_0.01/model_epoch-14_xe_model.pth
ensemble_model_4=models/soft_attention_inception_v4_seed_115_reconstruct_0.01/model_epoch-9_xe_model.pth


python3.6 image_caption_soft_att_infb_ense.py --lstm_size $lstm_size \
                                              --beam_file_path "$beam_file_path" \
                                              --ensemble_model_0 "$ensemble_model_0" \
                                              --ensemble_model_1 "$ensemble_model_1" \
                                              --ensemble_model_2 "$ensemble_model_2" \
                                              --ensemble_model_3 "$ensemble_model_3" \
                                              --ensemble_model_4 "$ensemble_model_4" \
                                              --beam_json_path "$beam_json_path"

echo $beam_file_path
echo $ensemble_model_0
echo $ensemble_model_1
echo $ensemble_model_2
echo $beam_json_path
