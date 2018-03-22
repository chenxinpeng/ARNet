#! encoding: UTF-8

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
from classLSTMCore import LSTMCore


class EncoderDecoder(nn.Module):
    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.lstm_size = opt.lstm_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)

        self.LSTMCore = LSTMCore(self.input_encoding_size, self.lstm_size, self.drop_prob_lm)

        # 注意因为 idx_to_word 是从 1 开始, 此处要加 1, 要不然会遇到 bug:
        # cuda runtime error (59) : device-side assert triggered
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.lstm_size, self.vocab_size)

        self.init_weights()

        # ----------------------------
        # 接着添加 reconstruct 参数部分
        # ----------------------------
        self.reconstruct_weights = opt.reconstruct_weight

        self.reconstruct_lstm = LSTMCore(self.input_encoding_size, self.lstm_size, self.drop_prob_lm)

        self.hidden_state_2_pre_hidden_state = nn.Linear(self.lstm_size, self.input_encoding_size)

        self.reconstruct_init_weights()

    def init_weights(self):
        initrange = 0.1
        self.img_embed.weight.data.uniform_(-initrange, initrange)
        self.img_embed.bias.data.fill_(0)

        self.embed.weight.data.uniform_(-initrange, initrange)

        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (Variable(weight.new(1, batch_size, self.lstm_size).zero_()),
                Variable(weight.new(1, batch_size, self.lstm_size).zero_()))

    # 初始化新添加的 reconstruct 部分
    def reconstruct_init_weights(self):
        initrange = 0.1
        self.hidden_state_2_pre_hidden_state.weight.data.uniform_(-initrange, initrange)
        self.hidden_state_2_pre_hidden_state.bias.data.fill_(0)

    def forward(self, fc_feats, seq):
        batch_size = fc_feats.size(0)

        state = self.init_hidden(batch_size)

        logit_seq = []
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                it = seq[:, i-1].clone()

                if seq[:, i-1].data.sum() == 0:
                    break

                xt = self.embed(it)

            output, state = self.LSTMCore.forward(xt, state)

            if i > 0:
                logit_words = F.log_softmax(self.logit(output.squeeze(0)))
                logit_seq.append(logit_words)

        return torch.cat([_.unsqueeze(1) for _ in logit_seq], 1).contiguous()

    def reconstruct_forward(self, fc_feats, seq, mask):
        batch_size = fc_feats.size(0)

        state = self.init_hidden(batch_size)
        reconstruct_state = (state[0].clone(), state[1].clone())

        logit_seq = []
        reconstructor_loss = 0.0

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)

                output, state = self.LSTMCore.forward(xt, state)

                previous_hidden_state = state[0].clone()

                reconstruct_output, reconstruct_state = self.reconstruct_lstm.forward(xt, reconstruct_state)
            else:
                it = seq[:, i-1].clone()

                if seq[:, i-1].data.sum() == 0:
                    break

                xt = self.embed(it)

                output, state = self.LSTMCore.forward(xt, state)

                logit_words = F.log_softmax(self.logit(output.squeeze(0)))
                logit_seq.append(logit_words)

                # 计算 reconstruct Loss
                reconstruct_output, reconstruct_state = self.reconstruct_lstm.forward(output, reconstruct_state)

                reconstruct_hidden_state = self.hidden_state_2_pre_hidden_state(reconstruct_output)
                reconstruct_mask = mask[:, i].contiguous().view(-1, batch_size).repeat(1, self.lstm_size)
                reconstruct_difference = reconstruct_hidden_state - previous_hidden_state

                current_reconstructor_loss = torch.sum(torch.sum(torch.mul(reconstruct_difference, reconstruct_difference) * reconstruct_mask, dim=1))
                current_reconstructor_loss = current_reconstructor_loss / batch_size * self.reconstruct_weights

                reconstructor_loss += current_reconstructor_loss

                previous_hidden_state = state[0].clone()

        return torch.cat([_.unsqueeze(1) for _ in logit_seq], 1).contiguous(), reconstructor_loss

    def sample_beam(self, fc_feats, init_index, opt={}):
        beam_size = opt.get('beam_size', 3)  # 如果不能取到 beam_size 这个变量, 则令 beam_size 为 3
        batch_size = fc_feats.size(0)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        top_seq = []
        top_prob = [[] for _ in range(batch_size)]
        done_beams = [[] for _ in range(batch_size)]

        for k in range(batch_size):
            state = self.init_hidden(beam_size)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam

            for t in range(self.seq_length + 1):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k+1]).expand(beam_size, self.input_encoding_size)

                elif t == 1:
                    it = fc_feats.data.new(beam_size).long().fill_(init_index)
                    xt = self.embed(Variable(it, requires_grad=False))

                else:
                    logprobsf = logprobs.float()
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size

                    if t == 2:  # at first time step only the first beam is active
                        rows = 1

                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c': ix.data[q, c],
                                               'q': q,
                                               'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})

                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 2:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-2].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-2].clone()

                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 2:
                            beam_seq[:t - 2, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 2, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t - 2, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 2, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                  'logps': beam_seq_logprobs[:, vix].clone(),
                                                  'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t - 2]
                    xt = self.embed(Variable(it.cuda()))

                if t >= 2:
                    state = new_state

                output, state = self.LSTMCore.forward(xt, state)

                logprobs = F.log_softmax(self.logit(output))

            done_beams[k] = sorted(done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = done_beams[k][0]['logps']

            # save result
            l = len(done_beams[k])
            top_seq_cur = torch.LongTensor(l, self.seq_length).zero_()

            for temp_index in range(l):
                top_seq_cur[temp_index] = done_beams[k][temp_index]['seq'].clone()
                top_prob[k].append(done_beams[k][temp_index]['p'])

            top_seq.append(top_seq_cur)

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), top_seq, top_prob

    def sample(self, fc_feats, init_index, opt={}):
        beam_size = opt.get('beam_size', 1)

        if beam_size > 1:
            return self.sample_beam(fc_feats, init_index, opt)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []
        logprobs_all = []

        state = self.init_hidden(batch_size)

        for t in range(self.seq_length):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:
                    # input BOS, 304
                    it = fc_feats.data.new(batch_size).long().fill_(init_index)
                else:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()

                xt = self.embed(Variable(it, requires_grad=False).cuda())

            if t >= 2:
                # stop when all finished
                if t == 2:
                    unfinished = it > 0
                else:
                    unfinished *= (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.LSTMCore.forward(xt, state)

            logprobs = F.log_softmax(self.logit(output))
            logprobs_all.append(logprobs)

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), \
               torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), \
               torch.cat([_.unsqueeze(1) for _ in logprobs_all], 1).contiguous()

    def teacher_forcing_get_hidden_states(self, fc_feats, seq):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                it = seq[:, i-1].clone()
                if seq[:, i-1].data.sum() == 0:
                    break
                xt = self.embed(it)
            output, state = self.LSTMCore.forward(xt, state)
            if i > 0:
                if batch_size == 1:
                    output = F.log_softmax(self.logit(output))
                else:
                    output = F.log_softmax(self.logit(output.squeeze(0)))
                outputs.append(output)

        return state[0], outputs

    def free_running_get_hidden_states(self, fc_feats, init_index, end_index):
        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []
        logprobs_all = []
        state = self.init_hidden(batch_size)

        for t in range(self.seq_length):
            if t == 0:
                xt = self.img_embed(fc_feats)
            if t == 1:
                it = fc_feats.data.new(batch_size).long().fill_(init_index)
                xt = self.embed(Variable(it, requires_grad=False).cuda())
            if t >= 2:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
                if it.cpu().numpy()[0] == end_index:
                    break
                xt = self.embed(Variable(it, requires_grad=False).cuda())
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.LSTMCore.forward(xt, state)
            logprobs = F.log_softmax(self.logit(output))
            logprobs_all.append(logprobs)

        return state[0], logprobs_all
