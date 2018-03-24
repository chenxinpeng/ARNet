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
from class_ende import *


def visilization(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = EncodeDecode(opt)
    model.copy_weights(opt.vis_model_path)
    model.cuda()

    test_data_index = list(range(len(test_data)))
    teacher_forcing_hidden_states = []
    free_running_hidden_states = []

    for start, end in zip(range(0, len(test_data_index), opt.vis_batch_size),
                          range(opt.vis_batch_size, len(test_data_index), opt.vis_batch_size)):
        print("start: {:3d}  end: {:3d}".format(start, end))

        current_code_matrix = []
        current_comment_matrix = []
        current_comment_mask = []
        current_batch_index = test_data_index[start:end]

        for idx in current_batch_index:
            current_code_matrix.append(train_data[idx]['code_matrix'])
            current_comment_matrix.append(train_data[idx]['comment_matrix'])
            current_comment_mask.append(train_data[idx]['comment_mask'])

        current_code_matrix = np.reshape(current_code_matrix, [-1, opt.code_truncate])
        current_comment_matrix = np.reshape(current_comment_matrix, [-1, opt.comment_truncate])
        current_comment_mask = np.reshape(current_comment_mask, [-1, opt.comment_truncate])

        current_code_matrix_cuda = Variable(torch.from_numpy(current_code_matrix), requires_grad=False).cuda()
        current_comment_matrix_cuda = Variable(torch.from_numpy(current_comment_matrix), requires_grad=False).cuda()
        current_comment_mask_cuda = Variable(torch.from_numpy(current_comment_mask), requires_grad=False).cuda()

        # training mode
        model.train()
        current_teacher_forcing_hidden_states = model.teacher_forcing_get_hidden_states(current_code_matrix_cuda,
                                                                                        current_comment_matrix_cuda,
                                                                                        current_comment_mask_cuda,
                                                                                        token2index['EOS'])

        # eval model
        model.eval()
        current_free_running_hidden_states = model.free_running_get_hidden_states(current_code_matrix_cuda,
                                                                                  token2index['BOS'], token2index['EOS'])

        # aggregate
        teacher_forcing_hidden_states.append(current_teacher_forcing_hidden_states.cpu().data.numpy())
        free_running_hidden_states.append(current_free_running_hidden_states.cpu().data.numpy())

    teacher_forcing_hidden_states = np.reshape(teacher_forcing_hidden_states, [-1, opt.lstm_size])
    free_running_hidden_states = np.reshape(free_running_hidden_states, [-1, opt.lstm_size])

    hidden_states = {'teacher_forcing': teacher_forcing_hidden_states,
                     'free_running': free_running_hidden_states}

    with open(opt.vis_save_path, 'wb') as f:
        cPickle.dump(hidden_states, f)

    compute_distance(hidden_states=hidden_states)


if __name__ == '__main__':
    opt = opts.parse_opt()
    visilization(opt)
