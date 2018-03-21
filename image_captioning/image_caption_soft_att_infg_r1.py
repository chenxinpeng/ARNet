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

from utils_rewards import *
from utils_model import *
from class_soft_att_r1 import *


def inference(opt, infer_images_names):
    model = ShowAttendTellModel(opt)
    model.copy_weights(opt.infer_model_path)
    model.cuda()
    model.eval()

    images_probs_gt = []
    images_probs_gd = []

    # infer_images_sents = []
    for idx, image_name in enumerate(infer_images_names):
        print("{},  {}".format(idx, image_name))

        img_gt_seq = train_images_captions_index[image_name][0]
        img_gt_seq = np.asarray(img_gt_seq).astype(np.int64).reshape(-1, img_gt_seq.shape[0])
        img_gt_seq_cuda = Variable(torch.from_numpy(img_gt_seq), requires_grad=False).cuda()

        img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
        img_feat_conv = np.reshape(img_feat_conv, [1, opt.conv_att_size, opt.conv_feat_size])

        img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
        img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

        img_feat_conv_cuda = Variable(torch.from_numpy(img_feat_conv), requires_grad=False).cuda()
        img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

        _, seq_gt, seq_probs_gt = model.forward(img_feat_conv_fc, img_feat_conv_cuda, img_gt_seq_cuda)
        _, seq_gd, seq_probs_gd = model.sample(img_feat_conv_fc, img_feat_conv_cuda, word_to_idx['BOS'], word_to_idx['EOS'])

        seq_gt_cpu = list(seq_gt.cpu().numpy().squeeze())
        seq_gd_cpu = list(seq_gd.cpu().numpy().squeeze())

        gen_seq_gt = index_to_sentence(seq_gt_cpu)
        gen_seq_gd = index_to_sentence(seq_gd_cpu)

        seq_probs_gt = list(seq_probs_gt.cpu().numpy().squeeze())
        seq_probs_gd = list(seq_probs_gd.cpu().numpy().squeeze())

        images_probs_gt.append(np.sum(seq_probs_gt))
        images_probs_gd.append(np.sum(seq_probs_gd))

        # infer_images_sents.append(img_gen_seq)
        print(gen_seq_gt, np.sum(seq_probs_gt))
        print(gen_seq_gd, np.sum(seq_probs_gd))
        print("\n")

    print("seq_probs_gt: {}".format(np.mean(images_probs_gt)))
    print("seq_probs_gd: {}".format(np.mean(images_probs_gt)))

    # _ = evaluate(opt.infer_json_path, infer_images_names, infer_images_sents)


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
