#! encoding: UTF-8

import os
import ipdb
from collections import OrderedDict

import torch

import opts
opt = opts.parse_opt()


# reconstruct LSTM 1st 需要添加下面的参数
'''
{'hidden_state_2_pre_hidden_state.weight',
 'hidden_state_2_pre_hidden_state.bias',
 'reconstruct_lstm.h2h.weight',
 'reconstruct_lstm.h2h.bias',
 'reconstruct_lstm.i2h.bias',
 'reconstruct_lstm.i2h.weight'}
'''

# 110: 20
# 111: 32
# 112: 23
# 113: 34
# 114: 34
# 115: 27
# 116: 33
# 117: 25

new_weights_vars = {}
initrange = 0.1

old_model_path = 'models/encoder_decoder_inception_v4_seed_110/model_epoch-20.pth'
new_model_path = 'models/encoder_decoder_inception_v4_seed_110/model_epoch-20_reconstruct_lstm_1st.pth'

old_model = torch.load(old_model_path)

new_weights_vars['reconstruct_lstm.h2h.weight'] = torch.Tensor(opt.input_encoding_size, 4 * opt.lstm_size).uniform_(-initrange, initrange).cuda()
new_weights_vars['reconstruct_lstm.h2h.bias'] = torch.Tensor(4 * opt.lstm_size).uniform_(0, 0).cuda()

new_weights_vars['reconstruct_lstm.i2h.weight'] = torch.Tensor(opt.input_encoding_size, 4 * opt.lstm_size).uniform_(-initrange, initrange).cuda()
new_weights_vars['reconstruct_lstm.i2h.bias'] = torch.Tensor(4 * opt.lstm_size).uniform_(0, 0).cuda()

new_weights_vars['hidden_state_2_pre_hidden_state.weight'] = torch.Tensor(opt.input_encoding_size, opt.lstm_size).uniform_(-initrange, initrange).cuda()
new_weights_vars['hidden_state_2_pre_hidden_state.bias'] = torch.Tensor(opt.lstm_size).uniform_(0, 0).cuda()

for key, var in old_model.items():
    new_weights_vars[key] = var

new_weights_vars = OrderedDict(new_weights_vars)

torch.save(new_weights_vars, new_model_path)
