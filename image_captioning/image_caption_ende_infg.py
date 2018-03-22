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
from class_ende import *


def inference(opt):
    infer_images_names = open(opt.infer_file_path, 'r').read().splitlines()

    model = EncoderDecoder(opt)
    model.load_state_dict(torch.load(opt.infer_model_path))
    model.cuda()
    model.eval()

    infer_images_sents = []
    for idx, image_name in enumerate(infer_images_names):
        print("{},  {}".format(idx, image_name))

        img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
        img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

        img_feat_conv_fc = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

        greedy_seq, greedy_seqLogprobs, greedy_logprobs_all = model.sample(img_feat_conv_fc, word_to_idx['BOS'], {'sample_max': 1})

        img_sent = index_to_sentence(list(greedy_seq.cpu().numpy().squeeze(0)))
        infer_images_sents.append(img_sent)
        print(img_sent + '\n')

    Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, SPICE = evaluate(opt.infer_json_path, infer_images_names, infer_images_sents)


if __name__ == '__main__':
    opt = opts.parse_opt()
    
    inference(opt)
