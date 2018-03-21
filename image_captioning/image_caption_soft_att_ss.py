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
from class_soft_att_ss import *


def train_xe(opt, train_images_names, val_images_names):
    if os.path.isdir(opt.xe_model_save_path) is False:
        os.mkdir(opt.xe_model_save_path)

    # make reproducible
    torch.manual_seed(opt.seed)
    if opt.use_cuda:
        torch.cuda.manual_seed(opt.seed)

    opt.vocab_size = len(idx_to_word)

    model = ShowAttendTellModel(opt)

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model_epoch-' + str(opt.start_from_epoch) + '.pth')))

    model.cuda()

    # training mode
    model.train()

    criterion = LanguageModelCriterion()

    current_learning_rate = opt.learning_rate
    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=current_learning_rate, betas=(opt.optim_alpha, opt.optim_beta), weight_decay=opt.optim_weight_decay)
    else:
        raise Exception("optim not supported: {}".format(opt.feature_type))

    # 也可以加载 optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer_epoch-' + str(opt.start_from_epoch) + '.pth')))
        set_lr(optimizer, opt.scst_learning_rate)

    max_CIDEr = 0.0
    max_CIDEr_epoch = 0
    iter_idx = 0

    for epoch in range(opt.max_epochs):
        random.shuffle(train_images_names)

        if epoch != 0 and epoch % 2 == 0:
            current_learning_rate *= 0.8
            set_lr(optimizer, current_learning_rate)

        # 增加的 scheduled sampling 部分
        frac = epoch // opt.scheduled_sampling_increase_every
        opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        model.ss_prob = opt.ss_prob

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
            current_masks = np.zeros((current_gt_sents.shape[0], current_gt_sents.shape[1]), dtype=np.float32)

            # in PY3, map is a generator, refer: https://stackoverflow.com/questions/44511752
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

            # 注意此处喂入的是与生成的句子错开的 label, 如 BOS 对应的是 a, 而不是 BOS
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

            iter_idx += 1
            time_end = time.time()
            print("iter {}  start {}  epoch {}  lr {:.10f}  ss {:.3f}  loss {:.3f}  time {:.3f}".format(iter_idx, start, epoch, current_learning_rate, model.ss_prob, train_loss, time_end - time_start))

        if np.mod(epoch, 1) == 0:
            print("epoch {} is done, saving the model ...".format(epoch))

            parameter_path = os.path.join(opt.xe_model_save_path, 'model_epoch-' + str(epoch) + '.pth')
            torch.save(model.state_dict(), parameter_path)
            print("parameter model saved to {}".format(parameter_path))

            optimizer_path = os.path.join(opt.xe_model_save_path, 'optimizer_epoch-' + str(epoch) + '.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            print("optimizer model saved to {}".format(optimizer_path))

            model.eval()

            val_images_sents = []
            for idx, image_name in enumerate(val_images_names):
                img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
                img_feat_conv = np.reshape(img_feat_conv, [-1, opt.conv_att_size, opt.conv_feat_size])
                img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
                img_feat_fc = np.reshape(img_feat_fc, [-1, opt.fc_feat_size])

                img_feat_conv_cuda = Variable(torch.from_numpy(img_feat_conv), requires_grad=False).cuda()
                img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

                greedy_seq, greedy_seq_seqLogprobs, greedy_seq_logprobs_all = model.sample(img_feat_conv_fc, img_feat_conv_cuda, word_to_idx['BOS'], {'sample_max': 1})

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


def train_scst(opt, train_images_names, val_images_names):
    if os.path.isdir(opt.scst_model_save_path) is False:
        os.mkdir(opt.scst_model_save_path)

    # make reproducible
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    opt.vocab_size = len(idx_to_word)

    model = ShowAttendTellModel(opt)

    model.load_state_dict(torch.load(opt.scst_base_model_path))

    model.cuda()

    # training mode
    model.train()

    scst_criterion = RewardCriterion()
    xe_criterion = LanguageModelCriterion()

    current_learning_rate = opt.scst_learning_rate
    if opt.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=current_learning_rate, betas=(opt.optim_alpha, opt.optim_beta), weight_decay=opt.optim_weight_decay)
    else:
        raise Exception("optim not supported: {}".format(opt.feature_type))

    # 也可以加载 optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer_epoch-' + str(opt.start_from_epoch) + '.pth')))
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
            # print("idx: {}  epoch: {}  lr: {:.10f}  avg_score: {:.3f}  avg_reward: {:.3f}  scst_loss: {:.3f}  xe_loss: {:.3f}  time: {:.3f}".format(iter_idx, epoch, current_learning_rate, avg_score, avg_reward, scst_loss_value, xe_loss_value, time_end - time_start))
            print("idx: {}  start:{}  epoch: {}  lr: {:.10f}  avg_score: {:.3f}  avg_reward: {:.3f}  scst_loss: {:.3f}  time: {:.3f}".format(iter_idx, start, epoch, current_learning_rate, avg_score, avg_reward, scst_loss_value, time_end - time_start))

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

                greedy_seq, greedy_seq_seqLogprobs, greedy_seq_logprobs_all = model.sample(img_feat_conv_fc, img_feat_conv_cuda, word_to_idx['BOS'], {'sample_max': 1})

                img_sent = index_to_sentence(list(greedy_seq.cpu().numpy().squeeze(0)))

                val_images_sents.append(img_sent)

            Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, SPICE = evaluate(opt.scst_train_json_path, val_images_names, val_images_sents)

            model.train()

            current_score = CIDEr

            if current_score >= max_score:
                max_score = current_score
                max_score_epoch = epoch

            print('current_score: {:.5f}  max_score: {:.5f}  max_score_epoch: {}'.format(current_score, max_score, max_score_epoch))

            if epoch - max_score_epoch > opt.early_stop_value:
                print('Metric score has no improvement, stop, max spice value: {}, max epoch: {}'.format(max_score, max_score_epoch))
                sys.exit(0)


def inference(opt, infer_images_names):
    model = ShowAttendTellModel(opt)

    model.load_state_dict(torch.load(opt.infer_model_path))

    model.cuda()

    # eval mode
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

        greedy_seq, greedy_seqLogprobs, greedy_logprobs_all = model.sample(img_feat_conv_fc, img_feat_conv_cuda, word_to_idx['BOS'], {'sample_max': 1})

        img_sent = index_to_sentence(list(greedy_seq.cpu().numpy().squeeze(0)))
        infer_images_sents.append(img_sent)
        print(img_sent + '\n')

    Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, SPICE = evaluate(opt.infer_json_path, infer_images_names, infer_images_sents)


def beam_search(opt, beam_images_names):
    model = ShowAttendTellModel(opt)

    model.load_state_dict(torch.load(opt.beam_model_path))

    model.cuda()

    # eval mode
    model.eval()

    beam_images_sents = []
    for idx, image_name in enumerate(beam_images_names):
        print("{},  {}".format(idx, image_name))

        img_feat_conv = np.load(os.path.join(opt.conv_feat_path, image_name + '.npy'))
        img_feat_conv = np.reshape(img_feat_conv, [1, opt.conv_att_size, opt.conv_feat_size])

        img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
        img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

        img_feat_conv_cuda = Variable(torch.from_numpy(img_feat_conv), requires_grad=False).cuda()
        img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

        best_sent, best_sent_Logprobs, top_sents, top_sents_Logprobs = model.sample(img_feat_conv_fc, img_feat_conv_cuda, word_to_idx['BOS'], {'beam_size': 3})

        img_sent_1st = index_to_sentence(list(best_sent.cpu().numpy().squeeze()))
        img_sent_2nd = index_to_sentence(list(top_sents[0][1].cpu().numpy().squeeze()))
        img_sent_3rd = index_to_sentence(list(top_sents[0][2].cpu().numpy().squeeze()))

        beam_images_sents.append(img_sent_1st)

        print(img_sent_1st + '\n')
        print(img_sent_2nd + '\n')
        print(img_sent_3rd + '\n')

    Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, SPICE = evaluate(opt.beam_json_path, beam_images_names, beam_images_sents)


def t_SNE_visilization(opt, train_images_names, val_images_names):
    # make reproducible
    torch.manual_seed(opt.seed)
    if opt.use_cuda:
        torch.cuda.manual_seed(opt.seed)

    opt.vocab_size = len(idx_to_word)
    train_images_names = sorted(train_images_names)
    val_images_names = sorted(val_images_names)

    model = ShowAttendTellModel(opt)
    model.load_state_dict(torch.load(opt.t_SNE_model_path))
    model.cuda()

    teacher_forcing_hidden_states = []
    free_running_hidden_states = []

    for start, end in zip(range(0, len(val_images_names), opt.t_SNE_batch_size),
                          range(opt.t_SNE_batch_size, len(val_images_names), opt.t_SNE_batch_size)):

        print("{}  {}".format(start, end))

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

        current_feats_conv_cuda = Variable(torch.from_numpy(current_feats_conv), requires_grad=False).cuda()
        current_feats_fc_cuda = Variable(torch.from_numpy(current_feats_fc), requires_grad=False).cuda()
        current_gt_sents_cuda = Variable(torch.from_numpy(current_gt_sents), requires_grad=False).cuda()

        # training mode
        model.train()
        current_teacher_forcing_hidden_states = model.teacher_forcing_get_hidden_states(current_feats_fc_cuda, current_feats_conv_cuda, current_gt_sents_cuda)

        # eval model
        model.eval()
        current_free_running_hidden_states = model.free_running_get_hidden_states(current_feats_fc_cuda, current_feats_conv_cuda, word_to_idx['BOS'])

        teacher_forcing_hidden_states.append(current_teacher_forcing_hidden_states.cpu().data.numpy())
        free_running_hidden_states.append(current_free_running_hidden_states.cpu().data.numpy())

    teacher_forcing_hidden_states = np.reshape(teacher_forcing_hidden_states, [-1, opt.lstm_size])
    free_running_hidden_states = np.reshape(free_running_hidden_states, [-1, opt.lstm_size])

    hidden_states = {'teacher_forcing': teacher_forcing_hidden_states, 'free_running': free_running_hidden_states}

    with open(opt.t_SNE_save_path, 'wb') as f:
        cPickle.dump(hidden_states, f)


if __name__ == '__main__':
    opt = opts.parse_opt()

    if opt.version == 'offline':
        train_images_names = open(opt.train_split, 'r').read().splitlines() + open(opt.restval_split, 'r').read().splitlines()
        val_images_names = open(opt.val_split, 'r').read().splitlines()
    elif opt.version == 'online':
        train_images_names = open(opt.train_split_online, 'r').read().splitlines()
        val_images_names = open(opt.val_split_online, 'r').read().splitlines()

    # MLE 训练
    if opt.function == 'train_xe':
        train_xe(opt, train_images_names, val_images_names)

    # SCST 阶段
    if opt.function == 'train_scst':
        train_scst(opt, train_images_names, val_images_names)

    # greedy inference
    if opt.function == 'inference':
        infer_images_names = open(opt.infer_file_path, 'r').read().splitlines()
        inference(opt, infer_images_names)

    # beam search
    if opt.function == 'beam_search':
        beam_images_names = open(opt.beam_file_path, 'r').read().splitlines()
        beam_search(opt, beam_images_names)

    if opt.function == 't_SNE_visilization':
        t_SNE_visilization(opt, train_images_names, val_images_names)
