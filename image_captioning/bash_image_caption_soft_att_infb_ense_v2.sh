#!/usr/bin/env bash

mode="back0"

export CUDA_VISIBLE_DEVICES=0

lstm_size=512

beam_file_path=data/splits/coco_test.txt
# beam_file_path=data/splits/submission_test_images.txt
beam_json_path=log/ensemble_beam_submission_test_images.json

# 0.735, 0.570, 0.434, 0.331, 0.260, 0.543, 1.023
ensemble_model_0=models/soft_attention_inception_v4_seed_117_reconstruct_2nd_0.005/model_epoch-1_xe_model.pth

# 0.737, 0.570, 0.435, 0.331, 0.259, 0.543, 1.020
ensemble_model_1=models/soft_attention_inception_v4_seed_117_reconstruct_0.01/model_epoch-7_xe_model.pth

# 0.738, 0.574, 0.437, 0.331, 0.258, 0.544, 1.021
ensemble_model_2=models/soft_attention_inception_v4_seed_112_reconstruct_2nd_0.05/model_epoch-3_xe_model.pth

# 0.740, 0.574, 0.438, 0.334, 0.260, 0.545, 1.032
ensemble_model_3=models/soft_attention_inception_v4_seed_117_h_512_rcst_v2_117_0.009/model_epoch-0.pth

# 0.740, 0.577, 0.440, 0.335, 0.259, 0.545, 1.034
ensemble_model_4=models/soft_attention_inception_v4_seed_117_h_512_rcst_v2_117_0.008/model_epoch-0.pth

# 0.740, 0.576, 0.439, 0.334, 0.260, 0.545, 1.031
ensemble_model_5=models/soft_attention_inception_v4_seed_117_h_512_rcst_v2_117_0.01/model_epoch-1.pth

# 0.741, 0.575, 0.439, 0.333, 0.258, 0.545, 1.028
ensemble_model_6=models/soft_attention_inception_v4_seed_117_h_512_rcst_v2_117_0.015/model_epoch-1.pth



# 0.740, 0.575, 0.438, 0.333, 0.259, 0.544, 1.030
ensemble_model_7=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_110_0.01/model_epoch-1.pth

# 0.741, 0.577, 0.441, 0.335, 0.260, 0.546, 1.036
ensemble_model_8=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_111_0.01/model_epoch-1.pth

# 0.741, 0.576, 0.441, 0.335, 0.259, 0.546, 1.035
ensemble_model_9=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_112_0.01/model_epoch-1.pth

# 0.739, 0.574, 0.438, 0.332, 0.259, 0.544, 1.028
ensemble_model_10=models/soft_att_inception_v4_seed_117_h_512_rcst_v2_113_0.01/model_epoch-0.pth



# 0.738, 0.573, 0.436, 0.330, 0.258, 0.544, 1.022
ensemble_model_11=models/soft_att_inception_v4_seed_115_h_512_rcst_v2_111_0.008/model_epoch-0.pth

# 0.739, 0.572, 0.434, 0.329, 0.257, 0.543, 1.019
ensemble_model_12=models/soft_att_inception_v4_seed_115_h_512_rcst_v2_112_0.008/model_epoch-0.pth

# 0.740, 0.573, 0.435, 0.330, 0.258, 0.544, 1.022
ensemble_model_13=models/soft_att_inception_v4_seed_115_h_512_rcst_v2_113_0.008/model_epoch-0.pth



# 0.739, 0.575, 0.439, 0.333, 0.258, 0.544, 1.021
ensemble_model_14=models/soft_att_inception_v4_seed_112_h_512_rcst_v2_112_0.01/model_epoch-0.pth

# 0.739, 0.574, 0.436, 0.330, 0.257, 0.543, 1.020
ensemble_model_15=models/soft_att_inception_v4_seed_112_h_512_rcst_v2_113_0.01/model_epoch-0.pth

# 0.738, 0.575, 0.439, 0.333, 0.257, 0.544, 1.018
ensemble_model_16=models/soft_att_inception_v4_seed_112_h_512_rcst_v2_110_0.01/model_epoch-0.pth



# 0.740, 0.575, 0.437, 0.331, 0.258, 0.544, 1.024
ensemble_model_17=models/soft_att_inception_v4_seed_115_h_512_rcst_v2_117_0.01/model_epoch-0.pth

# 0.738, 0.572, 0.435, 0.330, 0.257, 0.544, 1.021
ensemble_model_18=models/soft_att_inception_v4_seed_115_h_512_rcst_v2_115_0.008/model_epoch-0.pth



if [ "$mode" == "back" ]
then
    nohup python3.6 image_caption_soft_att_infb_ense_v2.py --lstm_size $lstm_size \
                                                           --beam_file_path "$beam_file_path" \
                                                           --beam_json_path "$beam_json_path" \
                                                           --ensemble_model_0 ${ensemble_model_0} \
                                                           --ensemble_model_1 ${ensemble_model_1} \
                                                           --ensemble_model_2 ${ensemble_model_2} \
                                                           --ensemble_model_3 ${ensemble_model_3} \
                                                           --ensemble_model_4 ${ensemble_model_4} \
                                                           --ensemble_model_5 ${ensemble_model_5} \
                                                           --ensemble_model_6 ${ensemble_model_6} \
                                                           --ensemble_model_7 ${ensemble_model_7} \
                                                           --ensemble_model_8 ${ensemble_model_8} \
                                                           --ensemble_model_9 ${ensemble_model_9} \
                                                           --ensemble_model_10 ${ensemble_model_10} \
                                                           --ensemble_model_11 ${ensemble_model_11} \
                                                           --ensemble_model_12 ${ensemble_model_12} \
                                                           --ensemble_model_13 ${ensemble_model_13} \
                                                           --ensemble_model_14 ${ensemble_model_14} \
                                                           --ensemble_model_15 ${ensemble_model_15} \
                                                           --ensemble_model_16 ${ensemble_model_16} \
                                                           --ensemble_model_17 ${ensemble_model_17} \
                                                           --ensemble_model_18 ${ensemble_model_18} \
                                                           > "${beam_json_path}".txt 2>&1 &
else
    python3.6 image_caption_soft_att_infb_ense_v2.py --lstm_size $lstm_size \
                                                     --beam_file_path "$beam_file_path" \
                                                     --beam_json_path "$beam_json_path" \
                                                     --ensemble_model_0 ${ensemble_model_0} \
                                                     --ensemble_model_1 ${ensemble_model_1} \
                                                     --ensemble_model_2 ${ensemble_model_2} \
                                                     --ensemble_model_3 ${ensemble_model_3} \
                                                     --ensemble_model_4 ${ensemble_model_4} \
                                                     --ensemble_model_5 ${ensemble_model_5} \
                                                     --ensemble_model_6 ${ensemble_model_6} \
                                                     --ensemble_model_7 ${ensemble_model_7} \
                                                     --ensemble_model_8 ${ensemble_model_8} \
                                                     --ensemble_model_9 ${ensemble_model_9} \
                                                     --ensemble_model_10 ${ensemble_model_10} \
                                                     --ensemble_model_11 ${ensemble_model_11} \
                                                     --ensemble_model_12 ${ensemble_model_12} \
                                                     --ensemble_model_13 ${ensemble_model_13} \
                                                     --ensemble_model_14 ${ensemble_model_14} \
                                                     --ensemble_model_15 ${ensemble_model_15} \
                                                     --ensemble_model_16 ${ensemble_model_16} \
                                                     --ensemble_model_17 ${ensemble_model_17} \
                                                     --ensemble_model_18 ${ensemble_model_18}
fi

echo ${beam_file_path}
echo ${ensemble_model_0}
echo ${ensemble_model_1}
echo ${ensemble_model_2}
echo ${ensemble_model_3}
echo ${ensemble_model_4}
echo ${ensemble_model_5}
echo ${ensemble_model_6}
echo ${ensemble_model_7}
echo ${ensemble_model_8}
echo ${ensemble_model_9}
echo ${ensemble_model_10}
echo ${ensemble_model_11}
echo ${ensemble_model_12}
echo ${ensemble_model_13}
echo ${ensemble_model_14}
echo ${ensemble_model_15}
echo ${ensemble_model_16}
echo ${ensemble_model_17}
echo ${ensemble_model_18}
echo ${beam_json_path}
