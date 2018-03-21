#! encoding: UTF-8

import os
import ipdb
import numpy as np
from six.moves import cPickle

import opts


def open_files(opt):
    with open(opt.train_images_captions_index, 'rb') as f:
        train_images_captions_index = cPickle.load(f, encoding="bytes")

    with open(opt.idx_to_word_path, 'rb') as f:
        idx_to_word = cPickle.load(f)

    return train_images_captions_index, idx_to_word

if __name__ == '__main__':
    opt = opts.parse_opt()
    train_images_captions_index, idx_to_word = open_files(opt)

    words_cnt = np.zeros([len(idx_to_word)], dtype=np.int32)

    for name, captions in train_images_captions_index.items():
        print(name)
        for i, sequence in enumerate(captions):
            for word_idx in sequence:
                if word_idx != 0:
                    words_cnt[word_idx-1] += 1

    ipdb.set_trace()
