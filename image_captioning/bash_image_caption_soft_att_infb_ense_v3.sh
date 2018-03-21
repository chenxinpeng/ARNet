#!/usr/bin/env bash

mode="back0"

export CUDA_VISIBLE_DEVICES=0

lstm_size=512

beam_file_path=data/splits/coco_test.txt
# beam_file_path=data/splits/submission_test_images.txt
beam_json_path=log/ensemble_beam_submission_test_images.json



# 0.740, 0.575, 0.438, 0.333, 0.259, 0.544, 1.030
ensemble_model_0=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_110_0.01/model_epoch-1.pth

# 0.741, 0.577, 0.441, 0.335, 0.260, 0.546, 1.036
ensemble_model_1=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_111_0.01/model_epoch-1.pth

# 0.741, 0.576, 0.441, 0.335, 0.259, 0.546, 1.035
ensemble_model_2=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_112_0.01/model_epoch-1.pth

# 0.739, 0.574, 0.438, 0.332, 0.259, 0.544, 1.028
ensemble_model_3=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_113_0.01/model_epoch-0.pth



if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_att_infb_ense_v3.py --lstm_size $lstm_size \
                                                           --beam_file_path "$beam_file_path" \
                                                           --beam_json_path "$beam_json_path" \
                                                           --ensemble_model_0 ${ensemble_model_0} \
                                                           --ensemble_model_1 ${ensemble_model_1} \
                                                           --ensemble_model_2 ${ensemble_model_2} \
                                                           --ensemble_model_3 ${ensemble_model_3} \
                                                           > "${beam_json_path}".txt 2>&1 &
else
    python3.6 image_caption_soft_att_infb_ense_v3.py --lstm_size $lstm_size \
                                                     --beam_file_path "$beam_file_path" \
                                                     --beam_json_path "$beam_json_path" \
                                                     --ensemble_model_0 ${ensemble_model_0} \
                                                     --ensemble_model_1 ${ensemble_model_1} \
                                                     --ensemble_model_2 ${ensemble_model_2} \
                                                     --ensemble_model_3 ${ensemble_model_3}
fi

echo ${beam_file_path}
echo ${ensemble_model_0}
echo ${ensemble_model_1}
echo ${ensemble_model_2}
echo ${ensemble_model_3}
echo ${beam_json_path}
