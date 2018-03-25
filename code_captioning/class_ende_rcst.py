from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from classLSTMCore import LSTMCore


class EncodeDecode(nn.Module):
    def __init__(self, opt):
        super(EncodeDecode, self).__init__()

        self.token_cnt = opt.token_cnt
        self.word_cnt = opt.word_cnt
        self.lstm_size = opt.lstm_size
        self.drop_prob = opt.drop_prob
        self.input_encoding_size = opt.input_encoding_size
        self.encode_time_step = opt.code_truncate
        self.decode_time_step = opt.comment_truncate

        self.encode_lstm = LSTMCore(self.input_encoding_size, self.lstm_size, self.drop_prob)
        self.decode_lstm = LSTMCore(self.input_encoding_size, self.lstm_size, self.drop_prob)

        self.embed = nn.Embedding(self.token_cnt + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.lstm_size, self.word_cnt)
        self.init_weights()

        # params of ARNet
        self.rcst_weight = opt.reconstruct_weight
        self.rcst_lstm = LSTMCore(self.lstm_size, self.lstm_size, self.drop_prob)
        self.h_2_pre_h = nn.Linear(self.lstm_size, self.lstm_size)
        self.rcst_init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(1, batch_size, self.lstm_size).zero_())
        init_c = Variable(weight.new(1, batch_size, self.lstm_size).zero_())
        init_state = (init_h, init_c)
        return init_state

    # init
    def rcst_init_weights(self):
        self.h_2_pre_h.weight.data.uniform_(-0.1, 0.1)
        self.h_2_pre_h.bias.data.fill_(0)

    # copy weights from pre-trained model
    def copy_weights(self, model_path):
        src_weights = torch.load(model_path)
        own_dict = self.state_dict()
        for key, var in src_weights.items():
            print("copy weights: {}  size: {}".format(key, var.size()))
            own_dict[key].copy_(var)

    def forward(self, code_matrix, comment_matrix, comment_mask):
        batch_size = code_matrix.size(0)
        encode_state = self.init_hidden(batch_size)
        decode_logit_seq = []

        # encoder
        for i in range(self.encode_time_step):
            encode_words = code_matrix[:, i].clone()

            if code_matrix[:, i].data.sum() == 0:
                break

            encode_xt = self.embed(encode_words)
            encode_output, encode_state = self.encode_lstm.forward(encode_xt, encode_state)

        # decoder
        decode_state = (encode_state[0].clone(), encode_state[1].clone())
        rcst_state = (encode_state[0].clone(), encode_state[1].clone())
        pre_h = encode_state[0].clone()
        rcst_loss = 0.0

        for i in range(self.decode_time_step):
            decode_words = comment_matrix[:, i].clone()

            if comment_matrix[:, i].data.sum() == 0:
                break

            decode_xt = self.embed(decode_words)
            decode_output, decode_state = self.decode_lstm.forward(decode_xt, decode_state)

            decode_logit_words = F.log_softmax(self.logit(decode_output))
            decode_logit_seq.append(decode_logit_words)

            # ARNet
            rcst_state, rcst_state = self.rcst_lstm.forward(decode_output, rcst_state)
            rcst_h = self.h_2_pre_h(rcst_state)

            rcst_diff = rcst_h - pre_h
            rcst_mask = comment_mask[:, i].contiguous().view(-1, batch_size).repeat(1, self.lstm_size)

            cur_rcst_loss = torch.sum(torch.sum(torch.mul(rcst_diff, rcst_diff) * rcst_mask, dim=1))
            rcst_loss += cur_rcst_loss * self.rcst_weight / torch.sum(comment_mask[:, i])

            # update previous hidden state
            pre_h = decode_state[0].clone()

        # aggregate
        decode_logit_seq = torch.cat([_.unsqueeze(1) for _ in decode_logit_seq], 1).contiguous()

        return decode_logit_seq, rcst_loss

    def sample(self, code_matrix, init_index, eos_index):
        batch_size = code_matrix.size(0)
        encode_state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []
        logprobs_all = []

        # encoder
        for i in range(self.encode_time_step):
            encode_words = code_matrix[:, i].clone()

            if code_matrix[:, i].data.sum() == 0:
                break

            encode_xt = self.embed(encode_words)
            encode_output, encode_state = self.encode_lstm.forward(encode_xt, encode_state)

        # decoder
        decode_state = (encode_state[0].clone(), encode_state[1].clone())
        for i in range(self.decode_time_step):
            if i == 0:
                it = code_matrix.data.new(batch_size).long().fill_(init_index)
                decode_xt = self.embed(Variable(it, requires_grad=False).cuda())
                decode_output, decode_state = self.decode_lstm.forward(decode_xt, decode_state)
            else:
                max_logprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()

                if it.sum() == eos_index:
                    break

                decode_xt = self.embed(Variable(it, requires_grad=False).cuda())
                decode_output, decode_state = self.decode_lstm.forward(decode_xt, decode_state)

                seq.append(it)
                seqLogprobs.append(max_logprobs.view(-1))

            logprobs = F.log_softmax(self.logit(decode_output))
            logprobs_all.append(logprobs)

        # aggregate
        greedy_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1).contiguous()
        greedy_seq_probs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1).contiguous()
        greedy_logprobs_all = torch.cat([_.unsqueeze(1) for _ in logprobs_all], 1).contiguous()

        return greedy_seq, greedy_seq_probs, greedy_logprobs_all
