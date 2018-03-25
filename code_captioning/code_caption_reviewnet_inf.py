from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import ipdb
import random

import torch
import torch.optim as optim
from torch.autograd import *

import opts
from utils_model import *
from class_reviewnet import *


def inference(opt):
    model = ReviewNet(opt)
    model.copy_weights(opt.test_model_path)
    model.cuda()
    model.eval()

    test_data_index = list(range(len(test_data)))
    opt.token_cnt = len(index2token)

    greedy_results = []
    gts_data = []

    for start, end in zip(range(0, len(test_data_index), 16),
                          range(16, len(test_data_index), 16)):
        print("start: {:3d}  end: {:3d}".format(start, end))

        current_code_matrix = []
        current_comment_matrix = []
        current_comment_mask = []
        current_comment_next = []

        current_batch_index = test_data_index[start:end]
        for idx in current_batch_index:
            current_code_matrix.append(val_data[idx]['code_matrix'])
            current_comment_matrix.append(val_data[idx]['comment_matrix'])
            current_comment_mask.append(val_data[idx]['comment_mask'])
            current_comment_next.append(val_data[idx]['comment_next'])

        current_code_matrix = np.reshape(current_code_matrix, [-1, opt.code_truncate])
        current_code_matrix_cuda = Variable(torch.from_numpy(current_code_matrix), requires_grad=False).cuda()
        current_comment_next = np.reshape(current_comment_next, [-1, opt.comment_truncate])

        greedy_seq, greedy_seqLogprobs, greedy_logprobs_all = model.sample(current_code_matrix_cuda,
                                                                           token2index['BOS'], token2index['EOS'])
        greedy_seq = greedy_seq.squeeze().cpu().numpy()

        for i in range(greedy_seq.shape[0]):
            greedy_results.append(greedy_seq[i])
            gts_data.append(current_comment_next[i])

    avg_score = get_scores(greedy_results, gts_data, token2index['BOS'], token2index['EOS'])
    avg_score_meteor = get_scores_meteor(greedy_results, gts_data, token2index['BOS'], token2index['EOS'])
    avg_score_cider = get_scores_cider(greedy_results, gts_data, token2index['BOS'], token2index['EOS'])
    avg_score_rouge = get_scores_rouge(greedy_results, gts_data, token2index['BOS'], token2index['EOS'])

    print("Bleu_1: {:.5f}  Bleu_2: {:.5f}  Bleu_3: {:.5f}  Bleu_4: {:.5f}  METEOR: {:.5f}  CIDEr: {:.5f}  ROUGE: {:.5f}".format(avg_score[0],
        avg_score[1], avg_score[2], avg_score[3], avg_score_meteor, avg_score_cider, avg_score_rouge))


if __name__ == '__main__':
    opt = opts.parse_opt()

    inference(opt)
