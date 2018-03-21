from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import ipdb
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable

sys.path.append("cider")
from cider.pyciderevalcap.ciderD.ciderD import CiderD
CiderD_scorer = CiderD(df='coco-train-idxs')

from spice.spice import Spice
SPICE_scorer = Spice()


def array_to_str(arr, end_word):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == end_word:
            break
    return out.strip()


def get_self_critical_reward(sample_result, greedy_result, gts_data, seq_per_img, init_word, end_word):
    sample_result = sample_result.cpu().numpy()
    greedy_result = greedy_result.cpu().numpy()
    batch_size = sample_result.shape[0]  # batch_size = sample_size * seq_per_img

    # res 前 0 ~ (batch-1) 为 sample 的句子, 后面 batch ~ (2 * batch-1) 是 greedy 的句子
    # 这里, 我自己又给 sample, greedy 的句子加上了 init_word('BOS') 单词
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [str(init_word) + ' ' + array_to_str(sample_result[i], end_word)]
    for i in range(batch_size):
        res[batch_size + i] = [str(init_word) + ' ' + array_to_str(greedy_result[i], end_word)]

    gts = OrderedDict()
    for i in range(len(gts_data)):
        gts[i] = [array_to_str(gts_data[i][j], end_word) for j in range(len(gts_data[i]))]

    # _, scores = Bleu(4).compute_score(gts, res)
    # scores = np.array(scores[3])
    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    avg_score, scores = CiderD_scorer.compute_score(gts, res)

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sample_result.shape[1], 1)

    return avg_score, rewards


# SPICE 需要知道具体的每个单词, 只用数字代表句子, 得出来的 SPICE 均为 0.0
def array_to_seq(arr, end_word, idx_to_word):
    out = ''
    for i in range(len(arr)):
        if arr[i] != 0:
            out += idx_to_word[arr[i]] + ' '
            if arr[i] == end_word:
                break
    return out.strip()


def get_self_critical_reward_spice(sample_result, greedy_result, gts_data, seq_per_img, init_word, end_word, idx_to_word):
    sample_result = sample_result.cpu().numpy()
    greedy_result = greedy_result.cpu().numpy()
    batch_size = sample_result.shape[0]  # batch_size = sample_size * seq_per_img

    # res 前 0 ~ (batch-1) 为 sample 的句子, 后面 batch ~ (2 * batch-1) 是 greedy 的句子
    # 这里, 我自己又给 sample, greedy 的句子加上了 init_word('BOS') 单词
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [idx_to_word[init_word] + ' ' + array_to_seq(sample_result[i], end_word, idx_to_word)]
    for i in range(batch_size):
        res[batch_size + i] = [idx_to_word[init_word] + ' ' + array_to_seq(greedy_result[i], end_word, idx_to_word)]

    gts = OrderedDict()
    for i in range(len(gts_data)):
        gts[i] = [array_to_seq(gts_data[i][j], end_word, idx_to_word) for j in range(len(gts_data[i]))]

    res = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    avg_score, scores = SPICE_scorer.compute_score(gts, res)

    scores = np.asarray(scores)

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], sample_result.shape[1], 1)

    return avg_score, rewards
