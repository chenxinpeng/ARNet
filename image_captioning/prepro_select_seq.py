from utils_model import *

import os
import ipdb
import json


if __name__ == "__main__":
    threshold = 14

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

        image_caption_len = len(image_caption.split(' '))

        if image_caption_len > threshold:
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

        image_caption_len = len(image_caption.split(' '))

        if image_caption_len > threshold:
            images_names_2.append(image_name)
            images_sents_2.append(image_caption)

    print(len(images_names_2))
    if len(images_names_2) > 0:
        _ = evaluate(eval_path_2, images_names_2, images_sents_2)
