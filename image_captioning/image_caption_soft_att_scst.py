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


def train_scst(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = ShowAttendTellModel(opt)
    model.load_state_dict(torch.load(opt.scst_base_model_path))
    model.cuda()
    model.train()

    scst_criterion = RewardCriterion()
    xe_criterion = LanguageModelCriterion()

    current_learning_rate = opt.scst_learning_rate
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
        set_lr(optimizer, opt.reconstruct_learning_rate)

    max_score = 0.0
    max_score_epoch = 0
    iter_idx = 0

    for epoch in range(opt.max_epochs):
        if epoch != 0 and epoch % 2 == 0:
            current_learning_rate *= 0.8
            set_lr(optimizer, current_learning_rate)

        for start, end in zip(range(0, len(train_images_names), int(opt.scst_batch_size)),
                              range(opt.scst_batch_size, len(train_images_names), opt.scst_batch_size)):

            time_start = time.time()

            # 等待当前设备上所有流中的所有核心完成
            torch.cuda.synchronize()

            current_feats_conv = []
            current_feats_fc = []
            # current_gt_sents = []
            gts_data = []

            image_names = train_images_names[start:end]
            for image_idx, image_name in enumerate(image_names):
                img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
                img_feat_conv = np.reshape(img_feat_conv, [opt.conv_att_size, opt.conv_feat_size])
                img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
                img_feat_fc = np.reshape(img_feat_fc, [opt.fc_feat_size])

                gts_data_tmp = []

                for i in range(opt.seq_per_img):
                    current_feats_conv.append(img_feat_conv)
                    current_feats_fc.append(img_feat_fc)
                    # current_gt_sents.append(train_images_captions_index[image_name][i])
                    gts_data_tmp.append(train_images_captions_index[image_name][i])

                gts_data_tmp = np.asarray(gts_data_tmp)
                gts_data.append(gts_data_tmp)

            current_feats_conv = np.reshape(current_feats_conv, [-1, opt.conv_att_size, opt.conv_feat_size])
            current_feats_fc = np.reshape(current_feats_fc, [-1, opt.fc_feat_size])

            # current_gt_sents = np.asarray(current_gt_sents).astype(np.int64)
            # current_gt_masks = np.zeros((current_gt_sents.shape[0], current_gt_sents.shape[1]), dtype=np.float32)

            # in PY3, map is a generator, refer: https://stackoverflow.com/questions/44511752
            # nonzeros = np.array(list(map(lambda x: (x != 0).sum(), current_gt_sents)))
            # for ind, row in enumerate(current_gt_masks):
            #    row[:nonzeros[ind]] = 1

            current_feats_conv_cuda = Variable(torch.from_numpy(current_feats_conv), requires_grad=False).cuda()
            current_feats_fc_cuda = Variable(torch.from_numpy(current_feats_fc), requires_grad=False).cuda()
            # current_gt_sents_cuda = Variable(torch.from_numpy(current_gt_sents), requires_grad=False).cuda()
            # current_gt_masks_cuda = Variable(torch.from_numpy(current_gt_masks), requires_grad=False).cuda()

            init_word = word_to_idx['BOS']
            end_word = word_to_idx['EOS']

            # 梯度归零
            optimizer.zero_grad()

            # 如果 {'sample_max': 0} 是 0, 则就是采样, 如果是 1, 则是 greedy decoding, 或者 beam search
            sample_seq, sample_seqLogprobs, sample_logprobs_all = model.sample(current_feats_fc_cuda, current_feats_conv_cuda, init_word, {'sample_max': 0})
            greedy_seq, greedy_seqLogprobs, greedy_logprobs_all = model.sample(current_feats_fc_cuda, current_feats_conv_cuda, init_word, {'sample_max': 1})

            avg_score, rewards = get_self_critical_reward(sample_seq, greedy_seq, gts_data, opt.seq_per_img, init_word, end_word)
            # avg_score, rewards = get_self_critical_reward_spice(sample_seq, greedy_seq, gts_data, opt.seq_per_img, init_word, end_word, idx_to_word)

            rewards_cuda = Variable(torch.from_numpy(rewards).float().cuda(), requires_grad=False)

            scst_loss = scst_criterion.forward(sample_seqLogprobs, sample_seq, rewards_cuda)

            # 同时加入很小的一个 XE 的 loss, 根据 paper:
            # A Deep Reinforced Model for Abstractive Summarization
            # xe_criterion_input = model.forward(current_feats_fc_cuda, current_feats_conv_cuda, current_gt_sents_cuda)
            # xe_loss = xe_criterion.forward(xe_criterion_input, current_gt_sents_cuda[:, 1:], current_gt_masks_cuda)

            # scst_loss *= (1 - opt.scst_loss_mixed_gamma)
            # xe_loss *= opt.scst_loss_mixed_gamma

            scst_loss.backward()
            # xe_loss.backward()

            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()

            torch.cuda.synchronize()

            scst_loss_value = scst_loss.data[0]
            # xe_loss_value = xe_loss.data[0]
            avg_reward = np.mean(rewards[:, 0])
            time_end = time.time()
            iter_idx += 1
            print("idx: {}  start:{}  epoch: {}  lr: {:.10f}  avg_score: {:.3f}  avg_reward: {:.3f}  scst_loss: {:.3f}  time: {:.3f}".format(iter_idx,
                start, epoch, current_learning_rate, avg_score, avg_reward, scst_loss_value, time_end - time_start))

        if np.mod(epoch, 1) == 0:
            print("epoch {} is done, saving the model ...".format(epoch))

            parameter_path = os.path.join(opt.scst_model_save_path, 'model_epoch-' + str(epoch) + '.pth')
            torch.save(model.state_dict(), parameter_path)
            print("parameter model saved to {}".format(parameter_path))

            optimizer_path = os.path.join(opt.scst_model_save_path, 'optimizer_epoch-' + str(epoch) + '.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            print("optimizer model saved to {}".format(optimizer_path))

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

            Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, SPICE = evaluate(opt.scst_train_json_path,
                                                                                     val_images_names,
                                                                                     val_images_sents)

            model.train()

            current_score = CIDEr

            if current_score >= max_score:
                max_score = current_score
                max_score_epoch = epoch

            print('current_score: {:.5f}  max_score: {:.5f}  max_score_epoch: {}'.format(current_score, max_score, max_score_epoch))

            if epoch - max_score_epoch > opt.early_stop_value:
                print('Metric score has no improvement, stop, max spice value: {}, max epoch: {}'.format(max_score, max_score_epoch))
                sys.exit(0)


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

    if os.path.isdir(opt.scst_model_save_path) is False:
        os.mkdir(opt.scst_model_save_path)

    train_scst(opt)
