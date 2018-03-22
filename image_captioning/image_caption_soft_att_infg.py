from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb
import random
from six.moves import cPickle

import torch
import torch.optim as optim
from torch.autograd import *

from utils_model import *
from class_soft_att import *


def inference(opt, infer_images_names):
    model = ShowAttendTellModel(opt)
    model.load_state_dict(torch.load(opt.infer_model_path))
    model.cuda()
    model.eval()

    infer_images_sents = []
    for idx, image_name in enumerate(infer_images_names):
        print("{},  {}".format(idx, image_name))

        img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
        img_feat_conv = np.reshape(img_feat_conv, [1, opt.conv_att_size, opt.conv_feat_size])

        img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
        img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

        img_feat_conv_cuda = Variable(torch.from_numpy(img_feat_conv), requires_grad=False).cuda()
        img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

        greedy_seq, greedy_seqLogprobs, greedy_logprobs_all = model.sample(img_feat_conv_fc, img_feat_conv_cuda,
                                                                           word_to_idx['BOS'], {'sample_max': 1})

        img_sent = index_to_sentence(list(greedy_seq.cpu().numpy().squeeze(0)))
        infer_images_sents.append(img_sent)
        print(img_sent + '\n')

    _ = evaluate(opt.infer_json_path, infer_images_names, infer_images_sents)


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

    infer_images_names = open(opt.infer_file_path, 'r').read().splitlines()

    inference(opt, infer_images_names)
