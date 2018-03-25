from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb
import math
import numpy as np
from collections import OrderedDict
from six.moves import cPickle

import torch
import torch.nn as nn

import opts
opt = opts.parse_opt()

from bleu.bleu import Bleu
from meteor.meteor import Meteor
from cider.cider import Cider
from rouge.rouge import Rouge
Bleu_score = Bleu(4)
Meteor_score = Meteor()
Cider_score = Cider()
Rouge_score = Rouge()


with open(opt.train_data_path, 'rb') as f:
    print('\nload  {}'.format(opt.train_data_path))
    train_data = cPickle.load(f)

with open(opt.val_data_path, 'rb') as f:
    print('\nload  {}'.format(opt.val_data_path))
    val_data = cPickle.load(f)

with open(opt.test_data_path, 'rb') as f:
    print('\nload  {}'.format(opt.test_data_path))
    test_data = cPickle.load(f)

with open(opt.token2index_path, 'rb') as f:
    print('\nload  {}'.format(opt.token2index_path))
    token2index = cPickle.load(f)

with open(opt.index2token_path, 'rb') as f:
    print('\nload  {}'.format(opt.index2token_path))
    index2token = cPickle.load(f)


# compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return 1.0 - sumxy / math.sqrt(sumxx * sumyy)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class EncodeDecodeCriterion(nn.Module):
    def __init__(self):
        super(EncodeDecodeCriterion, self).__init__()

    def forward(self, input, target, mask):
        batch_size = input.size(0)
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = - input.gather(1, target) * mask
        output = torch.sum(output) / batch_size

        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def array_to_str(arr, end_word):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == end_word:
            break
    return out.strip()


def get_scores(greedy_result, gts_data, init_word, end_word):
    batch_size = len(greedy_result)

    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [str(init_word) + ' ' + array_to_str(greedy_result[i], end_word)]

    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = [str(init_word) + ' ' + array_to_str(gts_data[i], end_word)]

    # _, scores = Bleu(4).compute_score(gts, res)
    # scores = np.array(scores[3])
    # res = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res = {i: res[i % batch_size] for i in range(batch_size)}
    gts = {i: gts[i % batch_size] for i in range(batch_size)}
    avg_score, scores = Bleu_score.compute_score(gts, res)

    return avg_score


def get_scores_meteor(greedy_result, gts_data, init_word, end_word):
    batch_size = len(greedy_result)

    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [str(init_word) + ' ' + array_to_str(greedy_result[i], end_word)]

    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = [str(init_word) + ' ' + array_to_str(gts_data[i], end_word)]

    res = {i: res[i % batch_size] for i in range(batch_size)}
    gts = {i: gts[i % batch_size] for i in range(batch_size)}
    avg_score, scores = Meteor_score.compute_score(gts, res)

    return avg_score


def get_scores_cider(greedy_result, gts_data, init_word, end_word):
    batch_size = len(greedy_result)

    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [str(init_word) + ' ' + array_to_str(greedy_result[i], end_word)]

    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = [str(init_word) + ' ' + array_to_str(gts_data[i], end_word)]

    res = {i: res[i % batch_size] for i in range(batch_size)}
    gts = {i: gts[i % batch_size] for i in range(batch_size)}
    avg_score, scores = Cider_score.compute_score(gts, res)

    return avg_score


def get_scores_rouge(greedy_result, gts_data, init_word, end_word):
    batch_size = len(greedy_result)

    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [str(init_word) + ' ' + array_to_str(greedy_result[i], end_word)]

    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = [str(init_word) + ' ' + array_to_str(gts_data[i], end_word)]

    res = {i: res[i % batch_size] for i in range(batch_size)}
    gts = {i: gts[i % batch_size] for i in range(batch_size)}
    avg_score, scores = Rouge_score.compute_score(gts, res)

    return avg_score


def compute_distance(hidden_states):
    teacher_forcing_hidden_states = np.squeeze(hidden_states['teacher_forcing'])
    free_running_hidden_states = np.squeeze(hidden_states['free_running'])

    shape_0 = free_running_hidden_states.shape[0]
    mean_teacher_forcing = np.mean(teacher_forcing_hidden_states, axis=0)
    mean_free_running = np.mean(free_running_hidden_states, axis=0)

    distance = np.sqrt(np.sum(np.square(mean_teacher_forcing - mean_free_running)))
    distance_2 = np.sqrt(np.sum(np.square(teacher_forcing_hidden_states - free_running_hidden_states), 1))

    print("distance_mc: {}".format(distance))
    print("distance_pw: {}".format(np.sum(distance_2)/shape_0))

    # cosine distance
    distance_cosine = 0.0
    for i in range(shape_0):
        current_tf_h = teacher_forcing_hidden_states[i]
        current_fr_h = free_running_hidden_states[i]
        distance_cosine += cosine_similarity(current_tf_h, current_fr_h)

    print("cosine distance_pw: {}".format(distance_cosine/shape_0))
    print("cosine distance_mc: {}".format(cosine_similarity(mean_teacher_forcing, mean_free_running)))
