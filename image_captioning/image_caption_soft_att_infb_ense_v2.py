from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import ipdb
import random
from six.moves import cPickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from utils_model import *
from class_soft_att_ense import *


def beam_search(opt, beam_images_names, beam_size=3):

    model_0 = ShowAttendTellModel(opt)
    model_1 = ShowAttendTellModel(opt)
    model_2 = ShowAttendTellModel(opt)
    model_3 = ShowAttendTellModel(opt)
    model_4 = ShowAttendTellModel(opt)
    model_5 = ShowAttendTellModel(opt)
    model_6 = ShowAttendTellModel(opt)
    model_7 = ShowAttendTellModel(opt)
    model_8 = ShowAttendTellModel(opt)
    model_9 = ShowAttendTellModel(opt)
    model_10 = ShowAttendTellModel(opt)
    model_11 = ShowAttendTellModel(opt)
    model_12 = ShowAttendTellModel(opt)
    model_13 = ShowAttendTellModel(opt)
    model_14 = ShowAttendTellModel(opt)
    model_15 = ShowAttendTellModel(opt)
    model_16 = ShowAttendTellModel(opt)
    model_17 = ShowAttendTellModel(opt)
    model_18 = ShowAttendTellModel(opt)

    model_0.copy_weights(opt.ensemble_model_0)
    model_1.copy_weights(opt.ensemble_model_1)
    model_2.copy_weights(opt.ensemble_model_2)
    model_3.copy_weights(opt.ensemble_model_3)
    model_4.copy_weights(opt.ensemble_model_4)
    model_5.copy_weights(opt.ensemble_model_5)
    model_6.copy_weights(opt.ensemble_model_6)
    model_7.copy_weights(opt.ensemble_model_7)
    model_8.copy_weights(opt.ensemble_model_8)
    model_9.copy_weights(opt.ensemble_model_9)
    model_10.copy_weights(opt.ensemble_model_10)
    model_11.copy_weights(opt.ensemble_model_11)
    model_12.copy_weights(opt.ensemble_model_12)
    model_13.copy_weights(opt.ensemble_model_13)
    model_14.copy_weights(opt.ensemble_model_14)
    model_15.copy_weights(opt.ensemble_model_15)
    model_16.copy_weights(opt.ensemble_model_16)
    model_17.copy_weights(opt.ensemble_model_17)
    model_18.copy_weights(opt.ensemble_model_18)

    model_0.cuda()
    model_1.cuda()
    model_2.cuda()
    model_3.cuda()
    model_4.cuda()
    model_5.cuda()
    model_6.cuda()
    model_7.cuda()
    model_8.cuda()
    model_9.cuda()
    model_10.cuda()
    model_11.cuda()
    model_12.cuda()
    model_13.cuda()
    model_14.cuda()
    model_15.cuda()
    model_16.cuda()
    model_17.cuda()
    model_18.cuda()

    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    model_8.eval()
    model_9.eval()
    model_10.eval()
    model_11.eval()
    model_12.eval()
    model_13.eval()
    model_14.eval()
    model_15.eval()
    model_16.eval()
    model_17.eval()
    model_18.eval()

    models = [model_0, model_1, model_2, model_3, model_4, model_5, model_6,
              model_7, model_8, model_9, model_10, model_11, model_12, model_13,
              model_14, model_15, model_16, model_17, model_18]

    beam_images_sents = []
    for idx, image_name in enumerate(beam_images_names):
        print("{},  {}".format(idx, image_name))

        img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
        img_feat_conv = np.reshape(img_feat_conv, [1, opt.conv_att_size, opt.conv_feat_size])

        img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
        img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

        img_feat_conv_cuda = Variable(torch.from_numpy(img_feat_conv), requires_grad=False).cuda()
        img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

        #############################################
        # ensemble
        #############################################
        seq = torch.LongTensor(opt.seq_length, 1).zero_()
        seqLogprobs = torch.FloatTensor(opt.seq_length, 1)
        top_seq = []
        top_prob = []
        done_beams = []
        beam_seq = torch.LongTensor(len(models), opt.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(len(models), opt.seq_length, beam_size).zero_()
        beam_logprobs_sum = torch.zeros(len(models), beam_size)  # running sum of logprobs for each beam

        att_feats_current = img_feat_conv_cuda.expand(beam_size, img_feat_conv_cuda.size(1), img_feat_conv_cuda.size(2))
        att_feats_current = att_feats_current.contiguous()

        states = []
        for t in range(opt.seq_length + 1):
            logits = []

            for j, model in enumerate(models):
                if t == 0:
                    a, b = model.get_init_state(img_feat_conv_fc, img_feat_conv_cuda, word_to_idx['BOS'], {'beam_size': 3})
                    logits.append(a)
                    states.append(b)
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""

                    # lets go to CPU for more efficiency in indexing operations
                    logprobsf = logprobs.float()

                    # sorted array of logprobs along each previous beam (last true = descending)
                    ys, ix = torch.sort(logprobsf, 1, True)

                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q, c]
                            candidate_logprob = beam_logprobs_sum[j][q] + local_logprob
                            if t > 1 and beam_seq[j][t - 2, q] == 0:
                                continue
                            candidates.append({'c': ix.data[q, c],
                                               'q': q,
                                               'p': candidate_logprob.data[0],
                                               'r': local_logprob.data[0]})
                    #if len(candidates) == 0:
                    #    break
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in states[j]]

                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[j][:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[j][:t - 1].clone()

                    for vix in range(min(beam_size, len(candidates))):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[j][:t - 1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[j][:t - 1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = states[j][state_ix][0, v['q']]  # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[j][t - 1, vix] = v['c']  # c'th word is the continuation
                        beam_seq_logprobs[j][t - 1, vix] = v['r']  # the raw logprob here
                        beam_logprobs_sum[j][vix] = v['p']  # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == opt.seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            done_beams.append({'seq': beam_seq[j][:, vix].clone(),
                                               'logps': beam_seq_logprobs[j][:, vix].clone(),
                                               'p': beam_logprobs_sum[j][vix]})

                    # encode as vectors
                    it = beam_seq[j][t - 1].clone()
                    xt = model.embed(Variable(it.cuda(), requires_grad=False))
                    state = new_state

                    output, state = model.core.forward(xt, att_feats_current, state)
                    logit = model.logit(output)
                    logits.append(logit)
                    states[j] = state

            # end of model
            logit = (logits[0] + logits[1] + logits[2] + logits[3] + logits[4] +
                     logits[5] + logits[6] + logits[7] + logits[8] + logits[9] +
                     logits[10] + logits[11] + logits[12] + logits[13] + logits[14] +
                     logits[15] + logits[16] + logits[17] + logits[18]) / float(len(models))
            logprobs = F.log_softmax(logit)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])
        seq[:, 0] = done_beams[0]['seq']  # the first beam has highest cumulative score
        seqLogprobs[:, 0] = done_beams[0]['logps']

        # save result
        l = len(done_beams)
        top_seq_cur = torch.LongTensor(l, opt.seq_length).zero_()

        for temp_index in range(l):
            top_seq_cur[temp_index] = done_beams[temp_index]['seq'].clone()
            top_prob.append(done_beams[temp_index]['p'])

        top_seq.append(top_seq_cur)

        best_sent = seq.transpose(0, 1)
        best_sent_Logprobs = seqLogprobs.transpose(0, 1)
        top_sents = top_seq
        top_sents_Logprobs = top_prob

        img_sent_1st = index_to_sentence(list(best_sent.cpu().numpy().squeeze()))
        img_sent_2nd = index_to_sentence(list(top_sents[0][1].cpu().numpy().squeeze()))
        img_sent_3rd = index_to_sentence(list(top_sents[0][2].cpu().numpy().squeeze()))

        beam_images_sents.append(img_sent_1st)

        print(img_sent_1st + '\n')
        print(img_sent_2nd + '\n')
        print(img_sent_3rd + '\n')

    _ = evaluate(opt.beam_json_path, beam_images_names, beam_images_sents)


if __name__ == '__main__':
    opt = opts.parse_opt()
    for var in vars(opt):
        print(var, getattr(opt, var))

    if opt.version == 'offline':
        train_images_names = open(opt.train_split, 'r').read().splitlines() + open(opt.restval_split, 'r').read().splitlines()
        val_images_names = open(opt.val_split, 'r').read().splitlines()

    elif opt.version == 'online':
        train_images_names = open(opt.train_split_online, 'r').read().splitlines()
        val_images_names = open(opt.val_split_online, 'r').read().splitlines()

    beam_images_names = open(opt.beam_file_path, 'r').read().splitlines()

    beam_search(opt, beam_images_names)
