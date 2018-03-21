#! encoding: UTF-8

import os
import ipdb
import glob
import numpy as np

old_conv_feat_path = '/data1/ailab_view/laviechen/image_caption_tf/inception/feats/train_val_test_feats_v4_conv'
old_fc_feat_path = '/data1/ailab_view/laviechen/image_caption_tf/inception/feats/train_val_test_feats_v4_fc'

new_conv_feat_path = '/data1/ailab_view/laviechen/image_caption_pytorch_me/data/feats/train_val_test_feats_v4_conv'
new_fc_feat_path = '/data1/ailab_view/laviechen/image_caption_pytorch_me/data/feats/train_val_test_feats_v4_fc'

conv_feat_lists = glob.glob(old_conv_feat_path + '/*.npy')
fc_feat_lists = glob.glob(old_fc_feat_path + '/*.npy')

for idx, image_name in enumerate(conv_feat_lists):
    print("{}  {}".format(idx, image_name))

    image_basename = os.path.basename(image_name)

    image_id = int(image_basename.split('.')[0].split('_')[2])

    current_conv_feat = np.load(image_name)
    current_fc_feat = np.load(os.path.join(old_fc_feat_path, image_basename))

    conv_feat_save_path = os.path.join(new_conv_feat_path, str(image_id) + '.npz')
    fc_feat_save_path = os.path.join(new_fc_feat_path, str(image_id) + '.npy')

    np.savez(conv_feat_save_path, feat=current_conv_feat)
    np.save(fc_feat_save_path, current_fc_feat)
