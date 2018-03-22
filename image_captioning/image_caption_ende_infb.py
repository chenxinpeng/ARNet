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


def beam_search(opt):
    beam_images_names = open(opt.beam_file_path, 'r').read().splitlines()

    model = EncoderDecoder(opt)
    model.load_state_dict(torch.load(opt.beam_model_path))
    model.cuda()
    model.eval()

    beam_images_sents = []
    for idx, image_name in enumerate(beam_images_names):
        print("{},  {}".format(idx, image_name))

        img_feat_fc = np.load(os.path.join(opt.fc_feat_path, image_name + '.npy'))
        img_feat_fc = np.reshape(img_feat_fc, [1, opt.fc_feat_size])

        img_feat_fc_cuda = Variable(torch.from_numpy(img_feat_fc), requires_grad=False).cuda()

        best_sent, best_sent_Logprobs, top_sents, top_sents_Logprobs = model.sample(img_feat_fc_cuda, word_to_idx['BOS'], {'beam_size': 3})

        img_sent_1st = index_to_sentence(list(best_sent.cpu().numpy().squeeze()))
        img_sent_2nd = index_to_sentence(list(top_sents[0][1].cpu().numpy().squeeze()))
        img_sent_3rd = index_to_sentence(list(top_sents[0][2].cpu().numpy().squeeze()))

        beam_images_sents.append(img_sent_1st)

        print(img_sent_1st + '\n')
        print(img_sent_2nd + '\n')
        print(img_sent_3rd + '\n')

    Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, SPICE = evaluate(opt.beam_json_path, beam_images_names, beam_images_sents)


if __name__ == '__main__':
    opt = opts.parse_opt()
    
    beam_search(opt)
