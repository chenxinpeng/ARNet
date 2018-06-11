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
import torch.optim as optim
from torch.autograd import *

from utils_model import *
from class_soft_att_rcst_lstm import *


def compute_distance(hidden_states):
    teacher_forcing_hidden_states = np.squeeze(hidden_states['teacher_forcing'])
    free_running_hidden_states = np.squeeze(hidden_states['free_running'])

    shape_0 = free_running_hidden_states.shape[0]
    print(shape_0)
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
        distance_cosine += cosine_distance(current_tf_h, current_fr_h)

    print("cosine distance_pw: {}".format(distance_cosine/shape_0))
    print("cosine distance_mc: {}".format(cosine_distance(mean_teacher_forcing, mean_free_running)))


def visilization(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = ShowAttendTellModel(opt)
    model.load_state_dict(torch.load(opt.vis_model_path))
    model.cuda()

    teacher_forcing_hidden_states = []
    free_running_hidden_states = []

    for start, end in zip(range(0, len(val_images_names), opt.vis_batch_size),
                          range(opt.vis_batch_size, len(val_images_names), opt.vis_batch_size)):
        print("{}  {}".format(start, end))

        image_names = val_images_names[start:end]

        current_feats_conv = []
        current_feats_fc = []
        current_gt_sents = []
        for image_name in image_names:
            img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
            img_feat_conv = np.reshape(img_feat_conv, [opt.conv_att_size, opt.conv_feat_size])
            img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
            img_feat_fc = np.reshape(img_feat_fc, [opt.fc_feat_size])

            current_feats_conv.append(img_feat_conv)
            current_feats_fc.append(img_feat_fc)
            current_gt_sents.append(train_images_captions_index[image_name][np.random.randint(0, 5)])

        current_feats_conv = np.reshape(current_feats_conv, [-1, opt.conv_att_size, opt.conv_feat_size])
        current_feats_fc = np.reshape(current_feats_fc, [-1, opt.fc_feat_size])
        current_gt_sents = np.asarray(current_gt_sents).astype(np.int64)

        current_feats_conv_cuda = Variable(torch.from_numpy(current_feats_conv), volatile=True).cuda()
        current_feats_fc_cuda = Variable(torch.from_numpy(current_feats_fc), volatile=True).cuda()
        current_gt_sents_cuda = Variable(torch.from_numpy(current_gt_sents), volatile=True).cuda()

        # training mode
        model.train()
        current_teacher_forcing_hidden_states, _ = model.teacher_forcing_get_hidden_states(current_feats_fc_cuda,
                                                                                           current_feats_conv_cuda,
                                                                                           current_gt_sents_cuda)

        # eval model
        model.eval()
        current_free_running_hidden_states, _ = model.free_running_get_hidden_states(current_feats_fc_cuda,
                                                                                     current_feats_conv_cuda,
                                                                                     word_to_idx['BOS'],
                                                                                     word_to_idx['EOS'])

        teacher_forcing_hidden_states.append(current_teacher_forcing_hidden_states.cpu().data.numpy())
        free_running_hidden_states.append(current_free_running_hidden_states.cpu().data.numpy())

    teacher_forcing_hidden_states = np.reshape(teacher_forcing_hidden_states, [-1, opt.lstm_size])
    free_running_hidden_states = np.reshape(free_running_hidden_states, [-1, opt.lstm_size])

    hidden_states = {'teacher_forcing': teacher_forcing_hidden_states, 'free_running': free_running_hidden_states}

    with open(opt.vis_save_path, 'wb') as f:
        cPickle.dump(hidden_states, f)

    compute_distance(hidden_states=hidden_states)


if __name__ == '__main__':
    opt = opts.parse_opt()
    for var in vars(opt):
        print(var, getattr(opt, var))

    opt.vocab_size = len(idx_to_word)

    if opt.version == 'offline':
        train_images_names = open(opt.train_split, 'r').read().splitlines() + open(opt.restval_split, 'r').read().splitlines()
        val_images_names = open(opt.val_split, 'r').read().splitlines()
    elif opt.version == 'online':
        train_images_names = open(opt.train_split_online, 'r').read().splitlines()
        val_images_names = open(opt.val_split_online, 'r').read().splitlines()

    train_images_names = sorted(train_images_names)
    val_images_names = sorted(val_images_names)

    visilization(opt)
