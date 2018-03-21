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

from utils_rewards import *
from utils_model import *
from class_soft_att_r1 import *


def evaluate_NLL(opt, train_images_names, val_images_names):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = ShowAttendTellModel(opt)
    model.load_state_dict(torch.load(opt.t_SNE_model_path))
    model.cuda()
    criterion = LanguageModelCriterion()

    nll_train = 0.0
    nll_inference = 0.0

    for start, end in zip(range(0, len(val_images_names)+1, opt.t_SNE_batch_size),
                          range(opt.t_SNE_batch_size, len(val_images_names)+1, opt.t_SNE_batch_size)):
        print("{}  {}".format(start, end))

        torch.cuda.synchronize()
        image_names = train_images_names[start:end]

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
        current_masks = np.zeros((current_gt_sents.shape[0], current_gt_sents.shape[1]), dtype=np.float32)

        # in PY3, map is a generator, refer: https://stackoverflow.com/questions/44511752
        nonzeros = np.array(list(map(lambda x: (x != 0).sum(), current_gt_sents)))
        for ind, row in enumerate(current_masks):
            row[:nonzeros[ind]] = 1

        current_feats_conv_cuda = Variable(torch.from_numpy(current_feats_conv), volatile=True).cuda()
        current_feats_fc_cuda = Variable(torch.from_numpy(current_feats_fc), volatile=True).cuda()
        current_gt_sents_cuda = Variable(torch.from_numpy(current_gt_sents), volatile=True).cuda()
        current_masks_cuda = Variable(torch.from_numpy(current_masks), volatile=True).cuda()
        
        # training mode
        model.train()
        criterion_input = model.forward(current_feats_fc_cuda, current_feats_conv_cuda, current_gt_sents_cuda)
        current_loss_train = criterion.forward(criterion_input, current_gt_sents_cuda[:, 1:], current_masks_cuda)

        # eval model
        model.eval()
        greedy_seq, greedy_seq_seqLogprobs, greedy_seq_logprobs_all = model.sample(current_feats_fc_cuda,
                                                                                   current_feats_conv_cuda,
                                                                                   word_to_idx['BOS'],
                                                                                   {'sample_max': 1})
        current_loss_inference = criterion.forward(greedy_seq_logprobs_all, greedy_seq, current_masks_cuda)

        print("train: {}  inference: {}".format(current_loss_train.data[0], current_loss_inference.data[0]))
        nll_train += current_loss_train.data[0]
        nll_inference += current_loss_inference.data[0]

    print(nll_train, nll_inference)


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

    opt.vocab_size = len(idx_to_word)
    train_images_names = sorted(train_images_names)
    val_images_names = sorted(val_images_names)

    evaluate_NLL(opt, train_images_names, val_images_names)
