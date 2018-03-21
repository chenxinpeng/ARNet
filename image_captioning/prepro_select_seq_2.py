#! encoding: UTF-8

from utils_model import *

import re
import os
import ipdb
import json


if __name__ == "__main__":
    threshold = 14
    threshold_sent_length = 12.5

    val_images_ids = []
    val_images_id2caption = {}

    path_original = 'data/annotations/captions_val2014.json'
    original_file = json.load(open(path_original, 'r'))

    for idx, image_info in enumerate(original_file['annotations']):
        image_id = image_info['image_id']
        image_caption = image_info['caption']
        if image_id not in val_images_id2caption.keys():
            val_images_id2caption[image_id] = []

        val_images_id2caption[image_id].append(image_caption)

    image_id2length = {}
    for image_id, image_captions in val_images_id2caption.items():
        words_cnt = 0
        for i, sent in enumerate(image_captions):
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

            words_cnt += len(tmp_sent)

        image_id2length[image_id] = words_cnt / 5

    image_ids_longer = []
    for image_id, image_length in image_id2length.items():
        if image_length > threshold_sent_length:
            image_ids_longer.append(image_id)

    path_1 = 'models/soft_attention_inception_v4_seed_117/model_epoch-8_beam_offline_test.json'
    path_2 = 'models/soft_attention_inception_v4_seed_117_reconstruct_0.0025/model_epoch-2_beam_offline_test.json'

    eval_path_1 = 'models/soft_attention_inception_v4_seed_117/tmp.json'
    eval_path_2 = 'models/soft_attention_inception_v4_seed_117_reconstruct_0.0025/tmp.json'

    train_val_imageIDs_to_imageNames = {}
    for k, v in train_val_imageNames_to_imageIDs.items():
        train_val_imageIDs_to_imageNames[v] = k

    json_file_1 = json.load(open(path_1, 'r'))
    json_file_2 = json.load(open(path_2, 'r'))

    images_names_1 = []
    images_sents_1 = []
    for idx, image in enumerate(json_file_1):
        image_id = image['image_id']
        image_name = train_val_imageIDs_to_imageNames[image_id]
        image_caption = image['caption']

        if image_id in image_ids_longer:
            images_names_1.append(image_name)
            images_sents_1.append(image_caption)

    print(len(images_names_1))
    if len(images_names_1) > 0:
        _ = evaluate(eval_path_1, images_names_1, images_sents_1)

    images_names_2 = []
    images_sents_2 = []
    for idx, image in enumerate(json_file_2):
        image_id = image['image_id']
        image_name = train_val_imageIDs_to_imageNames[image_id]
        image_caption = image['caption']

        if image_id in image_ids_longer:
            images_names_2.append(image_name)
            images_sents_2.append(image_caption)

    print(len(images_names_2))
    if len(images_names_2) > 0:
        _ = evaluate(eval_path_2, images_names_2, images_sents_2)
