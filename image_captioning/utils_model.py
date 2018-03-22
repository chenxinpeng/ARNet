from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from six.moves import cPickle

import os
import ipdb
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import opts
opt = opts.parse_opt()


# ==============================================================
# Load pre-processed data
# ==============================================================
with open(opt.idx_to_word_path, 'rb') as f:
    print('\nload  {}'.format(opt.idx_to_word_path))
    idx_to_word = cPickle.load(f)

with open(opt.word_to_idx_path, 'rb') as f:
    print('\nload  {}'.format(opt.word_to_idx_path))
    word_to_idx = cPickle.load(f)

with open(opt.train_images_captions_index, 'rb') as f:
    print('\nload  {}'.format(opt.train_images_captions_index))
    train_images_captions_index = cPickle.load(f, encoding="bytes")


def cosine_similarity(v1, v2):
    # compute cosine similarity of v1 to v2:
    # (v1 dot v2) / (||v1|| * ||v2||)
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return 1.0 - sumxy / math.sqrt(sumxx * sumyy)


# ===========================================
# function for generating sentence from index
# ===========================================
def index_to_sentence(sentences_index):
    sentences = []
    for idx_word in sentences_index:
        if idx_word == 0:
            break
        word = idx_to_word[idx_word]
        word = word.replace('\n', '').replace('\\', '').replace('"', '')
        sentences.append(word)

    punctuation = np.argmax(np.array(sentences) == 'EOS') + 1
    sentences = sentences[:punctuation]
    generated_sentence = ' '.join(sentences)
    generated_sentence = generated_sentence.replace('BOS ', '')
    generated_sentence = generated_sentence.replace(' EOS', '')

    return generated_sentence


# =============================================
# function for generating sentence from index
# 将句子反向之后的函数接口
# =============================================
def index_to_sentence_reverse(sentences_index):
    sentences = []
    for idx_word in sentences_index:
        word = idx_to_word[idx_word]
        word = word.replace('\n', '').replace('\\', '').replace('"', '')
        sentences.append(word)

    punctuation = np.argmax(np.array(sentences) == 'EOS') + 1
    sentences = sentences[:punctuation]
    generated_sentence = ' '.join(sentences)
    generated_sentence = generated_sentence.replace('BOS ', '')
    generated_sentence = generated_sentence.replace(' EOS', '')

    generated_sentence_reverse = generated_sentence.split(' ')
    generated_sentence_reverse.reverse()

    generated_sentence_reverse = ' '.join(generated_sentence_reverse)

    return generated_sentence_reverse


# ==================================================
# function for evaluate captions with COCO API
# 此处已经将 COCO 接口改写成 Python 3 的格式了
# ==================================================
def evaluate(json_path, images, captions, flag=True):
    import sys
    sys.path.append("coco-caption")
    annFile = 'data/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    fw = open(json_path, 'w')
    fw.write('[')
    for idx, image_name in enumerate(images):
        image_id = int(image_name.split('_')[2][:-4])
        caption = captions[idx]
        caption = caption.replace(' ,', ',').replace('"', '').replace('\n', '')
        if idx != len(images)-1:
            fw.write('{"image_id": ' + str(image_id) + ', "caption": "' + str(caption) + '"}, ')
        else:
            fw.write('{"image_id": ' + str(image_id) + ', "caption": "' + str(caption) + '"}]')
    fw.close()

    # coco evaluation
    coco = COCO(annFile)
    cocoRes = coco.loadRes(json_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    return cocoEval.eval['Bleu_1'], cocoEval.eval['Bleu_2'], cocoEval.eval['Bleu_3'], cocoEval.eval['Bleu_4'], \
           cocoEval.eval['CIDEr'], cocoEval.eval['METEOR'], cocoEval.eval['ROUGE_L'], cocoEval.eval['SPICE']


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt += ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        batch_size = input.size(0)

        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)

        mask_0 = (seq > 0).float()
        mask = Variable(to_contiguous(mask_0))
        mask = mask.view(-1)

        output = - input * reward * mask
        output = torch.sum(output) / batch_size

        return output


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        batch_size = input.size(0)

        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(2))

        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        target_cpu = target.data.cpu().numpy()
        if 10516 in target_cpu:
            ipdb.set_trace()

        output = - input.gather(1, target) * mask
        output = torch.sum(output) / batch_size

        return output


class ComputeFocalLoss(nn.Module):
    def __init__(self):
        super(ComputeFocalLoss, self).__init__()

    def forward(self, input, target, mask):
        batch_size = input.size(0)

        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(2))

        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = - input.gather(1, target) * mask
        output = torch.sum(output) / batch_size

        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)
