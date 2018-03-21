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


def train_xe(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = ShowAttendTellModel(opt)
    model.cuda()
    model.train()  # training mode
    criterion = LanguageModelCriterion()

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from,
                                                      'model_epoch-' + str(opt.start_from_epoch) + '.pth')))

    current_learning_rate = opt.learning_rate
    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=current_learning_rate,
                               betas=(opt.optim_alpha, opt.optim_beta),
                               weight_decay=opt.optim_weight_decay)
    else:
        raise Exception("optim not supported: {}".format(opt.feature_type))

    # 也可以加载 optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from,
                                                          'optimizer_epoch-' + str(opt.start_from_epoch) + '.pth')))
        set_lr(optimizer, opt.scst_learning_rate)

    max_CIDEr = 0.0
    max_CIDEr_epoch = 0

    for epoch in range(opt.max_epochs):
        random.shuffle(train_images_names)

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

            current_feats_conv = np.reshape(current_feats_conv, [opt.batch_size, opt.conv_att_size, opt.conv_feat_size])
            current_feats_fc = np.reshape(current_feats_fc, [opt.batch_size, opt.fc_feat_size])

            current_gt_sents = np.asarray(current_gt_sents).astype(np.int64)
            current_masks = np.zeros((current_gt_sents.shape[0], current_gt_sents.shape[1]), dtype=np.float32)

            # in PY3, map is a generator,
            # refer: https://stackoverflow.com/questions/44511752
            nonzeros = np.array(list(map(lambda x: (x != 0).sum(), current_gt_sents)))
            for ind, row in enumerate(current_masks):
                row[:nonzeros[ind]] = 1

            current_feats_conv_cuda = Variable(torch.from_numpy(current_feats_conv), requires_grad=False).cuda()
            current_feats_fc_cuda = Variable(torch.from_numpy(current_feats_fc), requires_grad=False).cuda()
            current_gt_sents_cuda = Variable(torch.from_numpy(current_gt_sents), requires_grad=False).cuda()
            current_masks_cuda = Variable(torch.from_numpy(current_masks), requires_grad=False).cuda()

            # 梯度归零
            optimizer.zero_grad()

            criterion_input = model.forward(current_feats_fc_cuda, current_feats_conv_cuda, current_gt_sents_cuda)

            # 注意此处喂入的是与生成的句子错开的 label,
            # 如 BOS 对应的是 a, 而不是 BOS
            loss = criterion.forward(criterion_input, current_gt_sents_cuda[:, 1:], current_masks_cuda)

            # 反向传播
            loss.backward()

            # XE 训练部分不进行 clip gradient
            # clip_gradient(optimizer, opt.grad_clip)

            # 更新参数
            optimizer.step()

            train_loss = loss.data[0]

            # 等待当前设备上所有流中的所有核心完成
            torch.cuda.synchronize()

            time_end = time.time()
            print("idx: {}  epoch: {}  lr:{:.8f}  loss: {:.3f}  time: {:.3f}".format(start, epoch, current_learning_rate,
                train_loss, time_end - time_start))

        if np.mod(epoch, 1) == 0:
            print("epoch {} is done, saving the model ...".format(epoch))

            parameter_path = os.path.join(opt.xe_model_save_path, 'model_epoch-' + str(epoch) + '.pth')
            torch.save(model.state_dict(), parameter_path)
            print("parameter model saved to {}".format(parameter_path))

            # optimizer_path = os.path.join(opt.xe_model_save_path, 'optimizer_epoch-' + str(epoch) + '.pth')
            # torch.save(optimizer.state_dict(), optimizer_path)
            # print("optimizer model saved to {}".format(optimizer_path))

            model.eval()

            val_images_sents = []
            for idx, image_name in enumerate(val_images_names):
                img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
                img_feat_conv = np.reshape(img_feat_conv, [1, opt.conv_att_size, opt.conv_feat_size])
                img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
                img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

                img_feat_conv_cuda = Variable(torch.from_numpy(img_feat_conv), requires_grad=False).cuda()
                img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

                greedy_seq, greedy_seq_seqLogprobs, greedy_seq_logprobs_all = model.sample(img_feat_conv_fc, img_feat_conv_cuda,  word_to_idx['BOS'], {'sample_max': 1})

                img_sent = index_to_sentence(list(greedy_seq.cpu().numpy().squeeze(0)))
                val_images_sents.append(img_sent)

            Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, SPICE = evaluate(opt.train_json_path, val_images_names, val_images_sents)

            model.train()

            if CIDEr >= max_CIDEr:
                max_CIDEr = CIDEr
                max_CIDEr_epoch = epoch
            print('current_CIDEr: {:.5f}  max_CIDEr: {:.5f}  max_CIDEr_epoch: {}'.format(CIDEr, max_CIDEr, max_CIDEr_epoch))
            if epoch - max_CIDEr_epoch > opt.early_stop_value:
                print('CIDEr has no improvement, stop. Max CIDEr value: {}, max epoch: {}'.format(max_CIDEr, max_CIDEr_epoch))
                sys.exit(0)


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

    if os.path.isdir(opt.xe_model_save_path) is False:
        os.mkdir(opt.xe_model_save_path)

    opt.vocab_size = len(idx_to_word)

    train_xe(opt)
