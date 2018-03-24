import os
import re
import ipdb
import opts
import numpy as np
from six.moves import cPickle


def data_string_split(seq):
    t = {}
    for idx, token in enumerate(seq.split(' ')):
        t[idx] = token
    return t


def data_prepare_data(args, filename, token2index, word_cnt):
    seq_data = {}

    data_lines = open(filename, 'r').read().splitlines()

    for idx, current_line in enumerate(data_lines):
        seq_data[idx] = {}
        # seq_data[idx]['code_matrix'] = []

        seq_0, seq_1 = current_line.split('\t')

        seq_0 = re.sub("\s+", " ", seq_0).strip()
        seq_1 = re.sub("\s+", " ", seq_1).strip()

        code_seq = seq_0.split(" ")
        comment_seq = seq_1.split(" ")

        # code_matrix: code seq
        # comment matrix: BOS + comment seq
        # comment next: comment seq + EOS
        # mask indices correspond to "matrix"
        code_matrix = np.zeros([args.code_truncate], dtype=np.int64)
        code_mask = np.zeros([args.code_truncate], dtype=np.float32)

        comment_matrix = np.zeros([args.comment_truncate], dtype=np.int64)
        comment_mask = np.zeros([args.comment_truncate], dtype=np.float32)
        comment_next = np.zeros([args.comment_truncate], dtype=np.int64)

        for i, token in enumerate(code_seq):
            if i < args.code_truncate:
                code_matrix[i] = token2index[token]
                code_mask[i] = 1
            else:
                break

        word_dict = {}
        word_label = np.zeros([word_cnt], dtype=np.int32)

        comment_matrix[0] = 2
        comment_mask[0] = 1
        for i, token in enumerate(comment_seq):
            if i < args.comment_truncate - 1:
                comment_matrix[i + 1] = token2index[token]
                comment_mask[i + 1] = 1
                comment_next[i] = token2index[token]
                comment_next[i + 1] = 1

                word_dict[comment_next[i]] = True

        ind = 0
        for key, val in word_dict.items():
            word_label[ind] = key
            ind += 1

        seq_data[idx]['code_matrix'] = code_matrix
        seq_data[idx]['code_mask'] = code_mask

        seq_data[idx]['comment_matrix'] = comment_matrix
        seq_data[idx]['comment_mask'] = comment_mask
        seq_data[idx]['comment_next'] = comment_next

        seq_data[idx]['word_label'] = word_label

    return seq_data


def data_indexing(args):
    global_data_lines = []
    global_data_lines.append(open(args.train_file_path, 'r').read().splitlines())
    global_data_lines.append(open(args.val_file_path, 'r').read().splitlines())
    global_data_lines.append(open(args.test_file_path, 'r').read().splitlines())

    code_set, comment_set = {}, {}

    for data_lines in global_data_lines:
        for idx, current_line in enumerate(data_lines):
            seq_0, seq_1 = current_line.split('\t')

            seq_0 = re.sub("\s+", " ", seq_0).strip()
            seq_1 = re.sub("\s+", " ", seq_1).strip()

            code_seq = data_string_split(seq_0)
            comment_seq = data_string_split(seq_1)

            for _, code_token in code_seq.items():
                code_set[code_token] = True

            for _, comment_token in comment_seq.items():
                comment_set[comment_token] = True

    token2index, index2token, token_cnt = {}, {}, 2

    # token2index['BOS'] = 1
    token2index['EOS'] = 1
    token2index['BOS'] = 2
    for _, token in enumerate(comment_set):
        token_cnt += 1
        token2index[token] = token_cnt

    word_cnt = token_cnt
    for _, token in enumerate(code_set):
        if token not in token2index.keys():
            token_cnt += 1
            token2index[token] = token_cnt

    for k, v in token2index.items():
        index2token[v] = k

    return token2index, index2token, token_cnt, word_cnt


if __name__ == '__main__':
    args = opts.parse_opt()

    token2index, index2token, token_cnt, word_cnt = data_indexing(args)
    print("\ntoken_cnt: {}".format(token_cnt))
    print("\nword_cnt: {}".format(word_cnt))

    with open(args.token2index_path, 'wb') as f:
        print("\nsaving {}".format(args.token2index_path))
        cPickle.dump(token2index, f)

    with open(args.index2token_path, 'wb') as f:
        print("\nsaving {}".format(args.index2token_path))
        cPickle.dump(index2token, f)

    train_data = data_prepare_data(args, args.train_file_path, token2index, word_cnt)
    val_data = data_prepare_data(args, args.val_file_path, token2index, word_cnt)
    test_data = data_prepare_data(args, args.test_file_path, token2index, word_cnt)

    with open(args.train_data_path, 'wb') as f:
        print("\nsaving {}".format(args.train_data_path))
        cPickle.dump(train_data, f)

    with open(args.val_data_path, 'wb') as f:
        print("\nsaving {}".format(args.val_data_path))
        cPickle.dump(val_data, f)

    with open(args.test_data_path, 'wb') as f:
        print("\nsaving {}".format(args.test_data_path))
        cPickle.dump(test_data, f)
