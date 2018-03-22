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

        self.fc2h = nn.Linear(self.fc_feat_size, self.lstm_size)

        self.core = LSTMSoftAttentionCore(self.input_encoding_size,
                                          self.lstm_size,
                                          self.conv_feat_size,
                                          self.conv_att_size,
                                          self.att_hidden_size,
                                          self.drop_prob_lm)

        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.lstm_size, self.vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc2h.weight.data.uniform_(-initrange, initrange)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)

        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)
        outputs = []

        for i in range(seq.size(1)-1):
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
            output = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
            outputs.append(output)

        vocab_log_probs = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()

        return vocab_log_probs  # e.g. batch * 19 * vocab_size

    def sample_beam(self, fc_feats, att_feats, init_index, opt={}):
        beam_size = opt.get('beam_size', 3)
        batch_size = fc_feats.size(0)
        fc_feat_size = fc_feats.size(1)

        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

        top_seq = []
        top_prob = [[] for _ in range(batch_size)]

        self.done_beams = [[] for _ in range(batch_size)]

        for k in range(batch_size):
            init_h = self.fc2h(fc_feats[k].unsqueeze(0).expand(beam_size, fc_feat_size))
            init_h = init_h.unsqueeze(0)
            init_c = init_h.clone()
            state = (init_h, init_c)

            att_feats_current = att_feats[k].unsqueeze(0).expand(beam_size, att_feats.size(1), att_feats.size(2))
            att_feats_current = att_feats_current.contiguous()

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam

            for t in range(self.seq_length + 1):
                if t == 0:
                    it = fc_feats.data.new(beam_size).long().fill_(init_index)
                    xt = self.embed(Variable(it, requires_grad=False))
                else:
                    # lets go to CPU for more efficiency in indexing operations
                    logprobsf = logprobs.float()

                    # sorted array of logprobs along each previous beam (last true = descending)
                    # ys: beam_size * (Vab_size + 1)
                    ys, ix = torch.sort(logprobsf, 1, True)

                    candidates = []

                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:  # at first time step only the first beam is active
                        rows = 1

                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            if t > 1 and beam_seq[t - 2, q] == 0:
                                continue
                            candidates.append({'c': ix.data[q, c],
                                               'q': q,
                                               'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})

                    if len(candidates) == 0:
                        break
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1].clone()

                    for vix in range(min(beam_size, len(candidates))):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(),
                                                       'logps': beam_seq_logprobs[:, vix].clone(),
                                                       'p': beam_logprobs_sum[vix]})

                    # encode as vectors
                    it = beam_seq[t - 1]
                    xt = self.embed(Variable(it.cuda(), requires_grad=False))
                    state = new_state

                output, state = self.core.forward(xt, att_feats_current, state)
                logprobs = F.log_softmax(self.logit(output), dim=1)

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

            # save result
            l = len(self.done_beams[k])
            top_seq_cur = torch.LongTensor(l, self.seq_length).zero_()

            for temp_index in range(l):
                top_seq_cur[temp_index] = self.done_beams[k][temp_index]['seq'].clone()
                top_prob[k].append(self.done_beams[k][temp_index]['p'])

            top_seq.append(top_seq_cur)

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), top_seq, top_prob

    def sample(self, fc_feats, att_feats, init_index, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, init_index, opt)

        batch_size = fc_feats.size(0)
        seq = []
        seqLogprobs = []
        logprobs_all = []

        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)

        for t in range(self.seq_length):
            if t == 0:
                it = fc_feats.data.new(batch_size).long().fill_(init_index)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()

                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False).cuda())  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False).cuda())

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished *= (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core.forward(xt, att_feats, state)

            logprobs = F.log_softmax(self.logit(output), dim=1)
            logprobs_all.append(logprobs)

        greedy_seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)
        greedy_seqLogprobs = torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)
        greedy_logprobs_all = torch.cat([_.unsqueeze(1) for _ in logprobs_all], 1).contiguous()

        return greedy_seq, greedy_seqLogprobs, greedy_logprobs_all

    def teacher_forcing_get_hidden_states(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)

        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)
        outputs = []

        for i in range(seq.size(1)):
            if i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
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
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core.forward(xt, att_feats, state)
            if batch_size == 1:
                output = F.log_softmax(self.logit(output), dim=1)
            else:
                output = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
            outputs.append(output)

        # return hidden states
        return state[0], outputs

    def free_running_get_hidden_states(self, fc_feats, att_feats, init_index, end_index):
        batch_size = fc_feats.size(0)
        logprobs_all = []

        init_h = self.fc2h(fc_feats)
        init_h = init_h.unsqueeze(0)
        init_c = init_h.clone()
        state = (init_h, init_c)

        for t in range(self.seq_length):
            if t == 0:  # input BOS
                it = fc_feats.data.new(batch_size).long().fill_(init_index)

            xt = self.embed(Variable(it, requires_grad=False))

            output, state = self.core.forward(xt, att_feats, state)

            if batch_size == 1:
                logprobs = F.log_softmax(self.logit(output), dim=1)
            else:
                logprobs = F.log_softmax(self.logit(output.squeeze(0)), dim=1)
            logprobs_all.append(logprobs)

            _, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
            if it.cpu().numpy()[0] == end_index:
                break

        return state[0], logprobs_all
