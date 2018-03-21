from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import ipdb
from six.moves import cPickle

import torch
from torch.autograd import *

from utils_model import *
from class_soft_att_rcst_v2 import *


def rcst(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = ShowAttendTellModel(opt)
    model.copy_weights(opt.rcst_model_path)
    model.cuda()
    model.train()
    criterion = LanguageModelCriterion()

    current_learning_rate = opt.rcst_learning_rate
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=current_learning_rate,
                                     betas=(opt.optim_alpha, opt.optim_beta),
                                     weight_decay=opt.optim_weight_decay)
    else:
        raise Exception("optim not supported: {}".format(opt.feature_type))

    max_CIDEr = 0.0
    max_CIDEr_epoch = 0

    for epoch in range(opt.max_epochs):
        if epoch != 0 and epoch % opt.learning_rate_decay_every == 0:
            current_learning_rate *= opt.learning_rate_decay_rate
            set_lr(optimizer, current_learning_rate)

        for start, end in zip(range(0, len(train_images_names), int(opt.batch_size/opt.seq_per_img)),
                              range(int(opt.batch_size/opt.seq_per_img), len(train_images_names), int(opt.batch_size/opt.seq_per_img))):

            time_start = time.time()

            # 等待当前设备上所有流中的所有核心完成
            torch.cuda.synchronize()

            current_feats_conv = []
            current_feats_fc = []
            current_gt_sents = []

            image_names = train_images_names[start:end]
            for image_name in image_names:
                img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
                img_feat_conv = np.reshape(img_feat_conv, [opt.conv_att_size, opt.conv_feat_size])

                img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
                img_feat_fc = np.reshape(img_feat_fc, [opt.fc_feat_size])

                for i in range(opt.seq_per_img):
                    current_feats_conv.append(img_feat_conv)
                    current_feats_fc.append(img_feat_fc)
                    current_gt_sents.append(train_images_captions_index[image_name][i])

            current_feats_conv = np.reshape(current_feats_conv, [-1, opt.conv_att_size, opt.conv_feat_size])
            current_feats_fc = np.reshape(current_feats_fc, [-1, opt.fc_feat_size])

            current_gt_sents = np.asarray(current_gt_sents).astype(np.int64)
            current_gt_masks = np.zeros((current_gt_sents.shape[0], current_gt_sents.shape[1]), dtype=np.float32)

            # in PY3, map is a generator, refer: https://stackoverflow.com/questions/44511752
            nonzeros = np.array(list(map(lambda x: (x != 0).sum(), current_gt_sents)))
            for ind, row in enumerate(current_gt_masks):
                row[:nonzeros[ind]] = 1

            current_feats_conv_cuda = Variable(torch.from_numpy(current_feats_conv), requires_grad=False).cuda()
            current_feats_fc_cuda = Variable(torch.from_numpy(current_feats_fc), requires_grad=False).cuda()
            current_gt_sents_cuda = Variable(torch.from_numpy(current_gt_sents), requires_grad=False).cuda()
            current_gt_masks_cuda = Variable(torch.from_numpy(current_gt_masks), requires_grad=False).cuda()

            # 梯度归零
            optimizer.zero_grad()

            # fc_feats, att_feats, seq, target, mask
            criterion_input, rcst_loss = model.rcst_forward(current_feats_fc_cuda, current_feats_conv_cuda,
                                                            current_gt_sents_cuda, current_gt_masks_cuda)
            xe_loss = criterion.forward(criterion_input, current_gt_sents_cuda[:, 1:], current_gt_masks_cuda)

            total_loss = xe_loss + rcst_loss

            # 反向传播
            total_loss.backward()

            # 更新参数
            optimizer.step()

            train_loss = total_loss.data[0]

            # 等待当前设备上所有流中的所有核心完成
            torch.cuda.synchronize()

            time_end = time.time()
            print("idx {}  epoch {}  lr: {:.10f}  total_loss: {:.3f}  xe_loss: {:.3f}  rcst_loss: {:.3f}  time: {:.3f}".format(start,
                epoch, current_learning_rate, train_loss, xe_loss.data[0], rcst_loss.data[0], time_end - time_start))

        if np.mod(epoch, 1) == 0:
            print("epoch {} is done, saving the model ...".format(epoch))

            parameter_path = os.path.join(opt.rcst_model_save_path, 'model_epoch-' + str(epoch) + '.pth')
            torch.save(model.state_dict(), parameter_path)
            print("parameter model saved to {}".format(parameter_path))

            optimizer_path = os.path.join(opt.rcst_model_save_path, 'optimizer_epoch-' + str(epoch) + '.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            print("optimizer model saved to {}".format(optimizer_path))

            # 把 model 设置为评估模式，只对 dropout 和 batch normalization 模块有影响
            model.eval()

            val_images_sents = []
            for idx, image_name in enumerate(val_images_names):
                img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
                img_feat_conv = np.reshape(img_feat_conv, [1, opt.conv_att_size, opt.conv_feat_size])
                img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
                img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

                img_feat_conv_cuda = Variable(torch.from_numpy(img_feat_conv), requires_grad=False).cuda()
                img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

                greedy_seq, greedy_seq_seqLogprobs, greedy_seq_logprobs_all = model.sample(img_feat_conv_fc,
                                                                                           img_feat_conv_cuda,
                                                                                           word_to_idx['BOS'],
                                                                                           {'sample_max': 1})

                img_sent = index_to_sentence(list(greedy_seq.cpu().numpy().squeeze(0)))
                val_images_sents.append(img_sent)

            _, _, _, _, CIDEr, _, _, _ = evaluate(opt.rcst_train_json_path, val_images_names, val_images_sents)

            if CIDEr >= max_CIDEr:
                max_CIDEr = CIDEr
                max_CIDEr_epoch = epoch
            print('current_CIDEr: {:.5f}  max_CIDEr: {:.5f}  max_CIDEr_epoch: {}'.format(CIDEr, max_CIDEr, max_CIDEr_epoch))
            if epoch - max_CIDEr_epoch > opt.early_stop_value:
                print('CIDEr has no improvement, stop. Max CIDEr value: {}, max epoch: {}'.format(max_CIDEr, max_CIDEr_epoch))
                sys.exit(0)

            model.train()


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

    if os.path.isdir(opt.rcst_model_save_path) is False:
        os.mkdir(opt.rcst_model_save_path)

    rcst(opt)
