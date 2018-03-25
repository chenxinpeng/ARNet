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
from torch.autograd import Variable

import opts
from utils_model import *
from class_ende_rcst import *


def ARNet(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    model = EncodeDecode(opt)

    # copy weights from pre-trained model with cross entropy
    model.copy_weights(opt.rcst_model_base_path)

    model.cuda()
    model.train()
    encode_decode_criterion = EncodeDecodeCriterion()

    current_learning_rate = opt.learning_rate
    optimizer = optim.Adam(model.parameters(),
                           lr=current_learning_rate,
                           betas=(opt.optim_alpha, opt.optim_beta),
                           weight_decay=opt.optim_weight_decay)

    train_data_index = list(range(len(train_data)))
    val_data_index = list(range(len(val_data)))
    iter_idx = 0

    for epoch in range(opt.max_epochs):
        random.shuffle(train_data_index)

        if epoch != 0 and epoch % opt.learning_rate_decay_every == 0:
            current_learning_rate *= opt.learning_rate_decay_rate
            set_lr(optimizer, current_learning_rate)

        for start, end in zip(range(0, len(train_data_index), opt.batch_size),
                              range(opt.batch_size, len(train_data_index), opt.batch_size)):

            time_start = time.time()

            # wait for synchronize
            torch.cuda.synchronize()

            current_code_matrix = []
            current_comment_matrix = []
            current_comment_mask = []
            current_comment_next = []

            current_batch_index = train_data_index[start:end]
            for idx in current_batch_index:
                current_code_matrix.append(train_data[idx]['code_matrix'])
                current_comment_matrix.append(train_data[idx]['comment_matrix'])
                current_comment_mask.append(train_data[idx]['comment_mask'])
                current_comment_next.append(train_data[idx]['comment_next'])

            current_code_matrix = np.reshape(current_code_matrix, [-1, opt.code_truncate])
            current_comment_matrix = np.reshape(current_comment_matrix, [-1, opt.comment_truncate])
            current_comment_mask = np.reshape(current_comment_mask, [-1, opt.comment_truncate])
            current_comment_next = np.reshape(current_comment_next, [-1, opt.comment_truncate])

            current_code_matrix_cuda = Variable(torch.from_numpy(current_code_matrix), requires_grad=False).cuda()
            current_comment_matrix_cuda = Variable(torch.from_numpy(current_comment_matrix), requires_grad=False).cuda()
            current_comment_mask_cuda = Variable(torch.from_numpy(current_comment_mask), requires_grad=False).cuda()
            current_comment_next_cuda = Variable(torch.from_numpy(current_comment_next), requires_grad=False).cuda()

            # setting gradients to zero
            optimizer.zero_grad()

            decode_logit_seq, rcst_loss = model.forward(current_code_matrix_cuda,
                                                        current_comment_matrix_cuda, current_comment_mask_cuda)
            encode_decode_loss = encode_decode_criterion.forward(decode_logit_seq,
                                                                 current_comment_next_cuda, current_comment_mask_cuda)

            # backward
            total_loss = encode_decode_loss + rcst_loss
            total_loss.backward()

            # update params
            optimizer.step()

            # wait for synchronize
            torch.cuda.synchronize()

            encode_decode_loss_val = encode_decode_loss.data[0]
            rcst_loss_val = rcst_loss.data[0]
            iter_idx += 1
            time_end = time.time()
            print("{}  {}  epoch: {}  lr: {:.8f}  encode_decode_loss: {:.3f}  rcst_loss: {:.3f}  time: {:.3f}".format(iter_idx,
                start, epoch, current_learning_rate, encode_decode_loss_val, rcst_loss_val, time_end - time_start))

        if np.mod(epoch, 1) == 0:
            print("\nepoch {} is done, saving the model ...".format(epoch))

            parameter_path = os.path.join(opt.rcst_model_save_path, 'model_epoch-' + str(epoch) + '.pth')
            torch.save(model.state_dict(), parameter_path)
            print("\nparameter model saved to {}".format(parameter_path))

            optimizer_path = os.path.join(opt.rcst_model_save_path, 'optimizer_epoch-' + str(epoch) + '.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            print("\noptimizer model saved to {}".format(optimizer_path))

            model.eval()

            greedy_results, gts_data = [], []
            for start, end in zip(range(0, len(val_data_index), opt.batch_size),
                                  range(opt.batch_size, len(val_data_index), opt.batch_size)):

                current_code_matrix = []
                current_comment_matrix = []
                current_comment_mask = []
                current_comment_next = []

                current_batch_index = val_data_index[start:end]
                for idx in current_batch_index:
                    current_code_matrix.append(val_data[idx]['code_matrix'])
                    current_comment_matrix.append(val_data[idx]['comment_matrix'])
                    current_comment_mask.append(val_data[idx]['comment_mask'])
                    current_comment_next.append(val_data[idx]['comment_next'])

                current_code_matrix = np.reshape(current_code_matrix, [-1, opt.code_truncate])
                current_code_matrix_cuda = Variable(torch.from_numpy(current_code_matrix), requires_grad=False).cuda()
                current_comment_next = np.reshape(current_comment_next, [-1, opt.comment_truncate])

                greedy_seq, greedy_seq_probs, greedy_logprobs_all = model.sample(current_code_matrix_cuda,
                                                                                 token2index['BOS'], token2index['EOS'])

                greedy_seq = greedy_seq.squeeze().cpu().numpy()

                for i in range(greedy_seq.shape[0]):
                    greedy_results.append(greedy_seq[i])
                    gts_data.append(current_comment_next[i])

            avg_score = get_scores(greedy_results, gts_data, token2index['BOS'], token2index['EOS'])

            print("current epoch: {}  Bleu_1: {:.5f}  Bleu_2: {:.5f}  Bleu_3: {:.5f}  Bleu_4: {:.5f}".format(epoch,
                avg_score[0], avg_score[1], avg_score[2], avg_score[3]))



if __name__ == '__main__':
    opt = opts.parse_opt()

    if os.path.isdir(opt.rcst_model_save_path) is False:
        os.mkdir(opt.rcst_model_save_path)

    ARNet(opt)
