from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import opts


class LSTMSoftAttentionCore(nn.Module):
    def __init__(self, input_encoding_size,
                 lstm_size,
                 conv_feat_size,
                 conv_att_size,
                 att_hidden_size,
                 drop_prob_lm):
        super(LSTMSoftAttentionCore, self).__init__()
        
        self.input_encoding_size = input_encoding_size
        self.lstm_size = lstm_size
        self.drop_prob_lm = drop_prob_lm
        self.conv_feat_size = conv_feat_size
        self.conv_att_size = conv_att_size
        self.att_hidden_size = att_hidden_size
        
        # build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 4 * self.lstm_size)
        self.h2h = nn.Linear(self.lstm_size, 4 * self.lstm_size)
        self.z2h = nn.Linear(self.conv_feat_size, 4 * self.lstm_size)
        
        # for soft attention
        self.att_2_att_h = nn.Linear(self.conv_feat_size, self.att_hidden_size)
        self.h_2_att_h = nn.Linear(self.lstm_size, self.att_hidden_size)
        self.att_h_2_out = nn.Linear(self.att_hidden_size, 1)
        
        self.dropout = nn.Dropout(self.drop_prob_lm)
        
        # init
        initrange = 0.1
        self.i2h.weight.data.uniform_(-initrange, initrange)
        self.i2h.bias.data.uniform_(-initrange, initrange)

        self.h2h.weight.data.uniform_(-initrange, initrange)
        self.h2h.bias.data.uniform_(-initrange, initrange)

        self.z2h.weight.data.uniform_(-initrange, initrange)
        self.z2h.bias.data.uniform_(-initrange, initrange)
        
        self.att_2_att_h.weight.data.uniform_(-initrange, initrange)
        self.att_2_att_h.bias.data.uniform_(-initrange, initrange)

        self.h_2_att_h.weight.data.uniform_(-initrange, initrange)
        self.h_2_att_h.bias.data.uniform_(-initrange, initrange)
        
        self.att_h_2_out.weight.data.uniform_(-initrange, initrange)
        self.att_h_2_out.bias.data.uniform_(-initrange, initrange)
        
    def forward(self, xt, att_seq, state):  # state = (pre_h, pre_c)
        pre_h = state[0][-1]
        pre_c = state[1][-1]

        # att_seq, batch * 64 * 1536
        att = att_seq.view(-1, self.conv_feat_size)
        att_linear = self.att_2_att_h(att)  # (batch * 64) * 512
        att_linear = att_linear.view(-1, self.conv_att_size, self.att_hidden_size)  # batch * 64 * 512

        # batch * 512
        # 64 * batch * 512 --> batch * 64 * 512
        h_linear = self.h_2_att_h(pre_h)
        h_linear_expand = h_linear.unsqueeze(0).expand(self.conv_att_size, h_linear.size(0), h_linear.size(1)).transpose(0, 1)

        att_h = F.tanh(h_linear_expand + att_linear)
        
        att_h_view = att_h.contiguous().view(-1, self.att_hidden_size)
        att_out = self.att_h_2_out(att_h_view).view(-1, self.conv_att_size)
        conv_weight = nn.Softmax(dim=1)(att_out)  # batch * conv_att_size
        conv_weight_unsqueeze = conv_weight.unsqueeze(2)  # batch * conv_att_size * 1

        # batch * conv_feat_size * conv_att_size, batch * 1536 * 64
        att_seq_t = att_seq.transpose(1, 2)

        # batch * conv_feat_size * 1 --> batch * conv_feat_size
        z = torch.bmm(att_seq_t, conv_weight_unsqueeze).squeeze(dim=2)

        all_input_sums = self.i2h(xt) + self.h2h(pre_h) + self.z2h(z)
        
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.lstm_size)
        sigmoid_chunk_sig = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk_sig.narrow(1, 0, self.lstm_size)
        forget_gate = sigmoid_chunk_sig.narrow(1, self.lstm_size, self.lstm_size)
        out_gate = sigmoid_chunk_sig.narrow(1, self.lstm_size * 2, self.lstm_size)

        in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.lstm_size, self.lstm_size))

        next_c = forget_gate * pre_c + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        next_h = self.dropout(next_h)

        output = next_h
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))

        return output, state

    # for ARNet
    def rcst_forward(self, xt, att_seq, state):  # state = (pre_h, pre_c)
        pre_h = state[0][-1]
        pre_c = state[1][-1]

        # att_seq, batch * 64 * 1536
        att = att_seq.view(-1, self.conv_feat_size)
        att_linear = self.att_2_att_h(att)  # (batch * 64) * 512
        att_linear = att_linear.view(-1, self.conv_att_size, self.att_hidden_size)  # batch * 64 * 512

        # batch * 512
        # 64 * batch * 512 --> batch * 64 * 512
        h_linear = self.h_2_att_h(pre_h)
        h_linear_expand = h_linear.unsqueeze(0).expand(self.conv_att_size, h_linear.size(0), h_linear.size(1)).transpose(0, 1)

        att_h = F.tanh(h_linear_expand + att_linear)

        att_h_view = att_h.contiguous().view(-1, self.att_hidden_size)
        att_out = self.att_h_2_out(att_h_view)
        att_out_view = att_out.view(-1, self.conv_att_size)
        conv_weight = nn.Softmax()(att_out_view)  # batch * conv_att_size
        conv_weight_unsqueeze = conv_weight.unsqueeze(2)  # batch * conv_att_size * 1

        att_seq_t = att_seq.transpose(1, 2)  # batch * conv_feat_size * conv_att_size, batch * 1536 * 64

        z = torch.bmm(att_seq_t, conv_weight_unsqueeze).squeeze(dim=2)  # batch * conv_feat_size * 1 --> batch * conv_feat_size

        all_input_sums = self.i2h(xt) + self.h2h(pre_h) + self.z2h(z)

        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.lstm_size)
        sigmoid_chunk_sig = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk_sig.narrow(1, 0, self.lstm_size)
        forget_gate = sigmoid_chunk_sig.narrow(1, self.lstm_size, self.lstm_size)
        out_gate = sigmoid_chunk_sig.narrow(1, self.lstm_size * 2, self.lstm_size)

        in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.lstm_size, self.lstm_size))

        next_c = forget_gate * pre_c + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        next_h = self.dropout(next_h)

        output = next_h
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))

        return output, state, z
