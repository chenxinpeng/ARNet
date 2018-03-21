#! encoding: UTF-8

import os
import ipdb
from collections import OrderedDict

import torch

import opts
opt = opts.parse_opt()

# 110: 12  reconstruct 0.005:  6  reconstruct 0.01:  7
# 111: 11  reconstruct 0.005:  1  reconstruct 0.01: 16
# 112: 10  reconstruct 0.005: 10  reconstruct 0.01:  7
# 113: 12  reconstruct 0.005:  4  reconstruct 0.01: 14
# 114:  9  reconstruct 0.005:  1  reconstruct 0.01:  6
# 115: 12  reconstruct 0.005:  5  reconstruct 0.01:  9
# 116: 14  reconstruct 0.005:  2  reconstruct 0.01: 10
# 117:  8  reconstruct 0.005: 18  reconstruct 0.01:  7

# 110: 12  reconstruct_2nd 0.005:  0  reconstruct_2nd 0.01:  0
# 111: 11  reconstruct_2nd 0.005:  1  reconstruct_2nd 0.01:  3
# 112: 10  reconstruct_2nd 0.005:  0  reconstruct_2nd 0.01:  0
# 113: 12  reconstruct_2nd 0.005:  1  reconstruct_2nd 0.01:  2
# 114:  9  reconstruct_2nd 0.005:  0  reconstruct_2nd 0.01:  2
# 115: 12  reconstruct_2nd 0.005:  0  reconstruct_2nd 0.01:  0
# 116: 14  reconstruct_2nd 0.005:  0  reconstruct_2nd 0.01:  3
# 117:  8  reconstruct_2nd 0.005:  1  reconstruct_2nd 0.01:  7

xe_model_weights = ["fc2h.weight",
                    "fc2h.bias",
                    "core.i2h.weight",
                    "core.i2h.bias",
                    "core.h2h.weight",
                    "core.h2h.bias",
                    "core.z2h.weight",
                    "core.z2h.bias",
                    "core.att_2_att_h.weight",
                    "core.att_2_att_h.bias",
                    "core.h_2_att_h.weight",
                    "core.h_2_att_h.bias",
                    "core.att_h_2_out.weight",
                    "core.att_h_2_out.bias",
                    "embed.weight",
                    "logit.weight",
                    "logit.bias"]

new_weights_vars = {}

old_model_path = 'models/soft_attention_inception_v4_seed_110_reconstruct_2nd_0.005/model_epoch-7.pth'
new_model_path = 'models/soft_attention_inception_v4_seed_110_reconstruct_2nd_0.005/model_epoch-7_xe_model.pth'

old_model = torch.load(old_model_path)

for key, var in old_model.items():
    if key in xe_model_weights:
        new_weights_vars[key] = var

new_weights_vars = OrderedDict(new_weights_vars)

torch.save(new_weights_vars, new_model_path)
