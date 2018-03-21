#! encoding: UTF-8

import os
import glob

from skimage import io
from skimage.transform import resize
from skimage.io import imsave


image_lists = glob.glob('mscoco/train2014/*.jpg') + \
              glob.glob('mscoco/val2014/*.jpg') + \
              glob.glob('mscoco/test2014/*.jpg')

index = 0

if index == 0:
    image_save_path = 'train_val_test_images_crop_top_left'
    if os.path.isdir(image_save_path) is False:
        os.mkdir(image_save_path)

elif index == 1:  # 截去右上
    image_save_path = 'train_val_test_images_crop_top_right'
    if os.path.isdir(image_save_path) is False:
        os.mkdir(image_save_path)

elif index == 2:  # 截去右下
    image_save_path = 'train_val_test_images_crop_bottom_right'
    if os.path.isdir(image_save_path) is False:
        os.mkdir(image_save_path)

elif index == 3:  # 截去左下
    image_save_path = 'train_val_test_images_crop_bottom_left'
    if os.path.isdir(image_save_path) is False:
        os.mkdir(image_save_path)

for idx, image_path in enumerate(image_lists):
    print("{}  {}".format(idx, image_path))

    image_name = os.path.basename(image_path)

    img_data = io.imread(image_path)

    img_height = img_data.shape[0]
    img_width = img_data.shape[1]

    if index == 0:  # 截去左上
        if len(img_data.shape) == 2:
            img_data_crop = img_data[int(img_height * 0.1):, int(img_width * 0.1):]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)
        else:
            img_data_crop = img_data[int(img_height * 0.1):, int(img_width * 0.1):, :]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)

    elif index == 1:  # 截去右上
        if len(img_data.shape) == 2:
            img_data_crop = img_data[int(img_height * 0.1):, :int(img_width * 0.9)]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)
        else:
            img_data_crop = img_data[int(img_height * 0.1):, :int(img_width * 0.9), :]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)

    elif index == 2:  # 截去右下
        if len(img_data.shape) == 2:
            img_data_crop = img_data[:int(img_height * 0.9), :int(img_width * 0.9)]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)
        else:
            img_data_crop = img_data[:int(img_height * 0.9), :int(img_width * 0.9), :]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)

    elif index == 3:  # 截去左下
        if len(img_data.shape) == 2:
            img_data_crop = img_data[:int(img_height * 0.9), int(img_width * 0.1):]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)
        else:
            img_data_crop = img_data[:int(img_height * 0.9), int(img_width * 0.1):, :]
            img_data_crop_resize = resize(img_data_crop, img_data.shape)

    imsave(os.path.join(image_save_path, image_name), img_data_crop_resize)
