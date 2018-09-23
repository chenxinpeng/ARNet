import os
import re
import numpy as np
from six.moves import cPickle
import time
import opts


# -----------------------------------------------------------------
# Borrowed this function from NeuralTalk:
# https://github.com/karpathy/neuraltalk/blob/master/driver.py#L16
# -----------------------------------------------------------------
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print('Preprocessing word counts and creating vocab based on word count threshold %d' % word_count_threshold)
    t0 = time.time()

    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        sent = sent.lower()
        sent = sent.replace(',', ' ,')
        sent = sent.replace('\n', '').replace('"', '')
        sent = sent.replace('.', '').replace('?', '').replace('!', '')
        sent = sent.replace('``', '').replace('`', '').replace("''", '')
        sent = sent.replace(':', '').replace('-', '').replace('--', '')
        sent = sent.replace('...', '').replace(';', '')
        sent = sent.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        sent = sent.replace('@', '').replace('#', '').replace('$', '').replace('&', '').replace('*', '')
        sent = sent.replace('\\', '').replace('/', '')
        sent = sent.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '')
        sent = sent.replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace('10', '')

        sent = 'BOS ' + sent + ' EOS'
        sent = re.sub('\s+', ' ', sent).strip()
        tmp_sent = sent.split(' ')

        if ' ' in tmp_sent: tmp_sent.remove(' ')
        if '\n' in tmp_sent: tmp_sent.remove('\n')
        if '"' in tmp_sent: tmp_sent.remove('"')

        for w in tmp_sent:
           word_counts[w] = word_counts.get(w, 0) + 1

    # calculate the number of word, UNK
    unk_count = 0
    for w, c in word_counts.items():
        if c < word_count_threshold:
            unk_count = unk_count + c
    word_counts['UNK'] = unk_count

    # filter the word less than the threshold
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('Filter words from %d to %d in %0.2fs' % (len(word_counts), len(vocab), time.time()-t0))

    ixtoword = {}
    wordtoix = {}
    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+1
        ixtoword[idx+1] = w

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector


# -------------------------------------------
# generate mapping between words and indices
# -------------------------------------------
def generate_train_index(images_captions):
    print("change the word of each image captions to index by word_to_idx ...")
    count = 0
    train_images_captions_index = {}

    for each_img, sents in images_captions.items():
        sents_index = np.zeros([len(sents), opt.lstm_step], dtype=np.int32)
        for idy, sent in enumerate(sents):
            sent = sent.lower()
            sent = sent.replace(',', ' ,')
            sent = sent.replace('\n', '').replace('"', '')
            sent = sent.replace('.', '').replace('?', '').replace('!', '')
            sent = sent.replace('``', '').replace('`', '').replace("''", '')
            sent = sent.replace(':', '').replace('-', '').replace('--', '')
            sent = sent.replace('...', '').replace(';', '')
            sent = sent.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
            sent = sent.replace('@', '').replace('#', '').replace('$', '').replace('&', '').replace('*', '')
            sent = sent.replace('\\', '').replace('/', '')
            sent = sent.replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '')
            sent = sent.replace('6', '').replace('7', '').replace('8', '').replace('9', '').replace('0', '')

            sent = 'BOS ' + sent + ' EOS'
            sent = re.sub('\s+', ' ', sent).strip()
            tmp_sent = sent.split(' ')

            if ' ' in tmp_sent: tmp_sent.remove(' ')
            if '\n' in tmp_sent: tmp_sent.remove('\n')
            if '"' in tmp_sent: tmp_sent.remove('"')

            for idx, word in enumerate(tmp_sent):
                if idx == opt.lstm_step-1:
                    sents_index[idy, idx] = word_to_idx['EOS']
                    break
                if word in word_to_idx:
                    sents_index[idy, idx] = word_to_idx[word]
                if word not in word_to_idx:
                    sents_index[idy, idx] = word_to_idx["UNK"]

        train_images_captions_index[each_img] = sents_index
        count += 1
        print("{}  {}  {}".format(count, each_img, len(sents)))
    return train_images_captions_index


if __name__ == "__main__":
    opt = opts.parse_opt()

    with open(opt.official_train_captions_path, 'rb') as train_fr:
        train_images_captions = cPickle.load(train_fr)

    with open(opt.official_val_captions_path, 'rb') as val_fr:
        val_images_captions = cPickle.load(val_fr)

    # combine all sentences in captions
    all_sents = []
    for image, sents in train_images_captions.items():
        for each_sent in sents:
            all_sents.append(each_sent)

    for image, sents in val_images_captions.items():
        for each_sent in sents:
            all_sents.append(each_sent)

    word_to_idx, idx_to_word, bias_init_vector = preProBuildWordVocab(all_sents, word_count_threshold=5)

    images_captions = dict(train_images_captions, **val_images_captions)
    train_images_captions_index = generate_train_index(images_captions)

    # save
    with open(opt.idx_to_word_path, 'wb') as fw:
        cPickle.dump(idx_to_word, fw)

    with open(opt.word_to_idx_path, 'wb') as fw:
        cPickle.dump(word_to_idx, fw)

    np.save(opt.bias_init_vector_path, bias_init_vector)

    with open(opt.train_images_captions_index, 'wb') as f:
        cPickle.dump(train_images_captions_index, f)
