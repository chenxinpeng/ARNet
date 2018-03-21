from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import opts
from classLSTMSoftAttCore import LSTMSoftAttentionCore


class ShowAttendTellModel(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.lstm_size = opt.lstm_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length

        self.fc_feat_size = opt.fc_feat_size
        self.conv_feat_size = opt.conv_feat_size
        self.conv_att_size = opt.conv_att_size
        self.att_hidden_size = opt.att_hidden_size

        self.use_cuda = opt.use_cuda

        # Schedule sampling probability
        self.ss_prob = 0.0

        # 由 fc feature 初始化 LSTM 的 state, (c, h)
        self.fc2h = nn.Linear(self.fc_feat_size, self.lstm_size)

        self.core = LSTMSoftAttentionCore(self.input_encoding_size,
                                          self.lstm_size,
                                          self.conv_feat_size,
                                          self.conv_att_size,
                                          self.att_hidden_size,
                                          self.drop_prob_lm)

        # 注意因为 idx_to_word 是从 1 开始, 此处要加 1, 要不然会遇到 bug:
        # cuda runtime error (59) : device-side assert triggered
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.lstm_size, self.vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc2h.weight.data.uniform_(-initrange, initrange)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)

    def copy_weights(self, model_path):
        src_weights = torch.load(model_path)
        own_dict = self.state_dict()
        for key, var in own_dict.items():
            print("copy weights: {}  size: {}".format(key, var.size()))
            own_dict[key].copy_(src_weights[key])

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)

        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)
        outputs = []

        # 此处减 1 是考虑到需要最后一个作为 label
        for i in range(seq.size(1)-1):
            # scheduled sampling
            if i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()

            # break if all the sequences end
            if seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core.forward(xt, att_feats, state)
            output = F.log_softmax(self.logit(output.squeeze(0)))
            outputs.append(output)

        vocab_log_probs = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

        return vocab_log_probs

    def get_init_state(self, fc_feats, att_feats, init_index, opt={}):
        beam_size = opt.get('beam_size', 10)

        init_h = self.fc2h(fc_feats.expand(beam_size, fc_feats.size(1)))
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)

        att_feats_current = att_feats.expand(beam_size, att_feats.size(1), att_feats.size(2))
        att_feats_current = att_feats_current.contiguous()

        it = fc_feats.data.new(beam_size).long().fill_(init_index)
        xt = self.embed(Variable(it, requires_grad=False))

        output, state = self.core.forward(xt, att_feats_current, state)
        logits = self.logit(output)

        return logits, state
