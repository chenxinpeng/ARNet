import argparse
import sys


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--function', type=str, default='train_with_mle')
    parser.add_argument('--version', type=str, default='offline')

    parser.add_argument('--train_annotations', type=str, default='data/annotations/captions_train2014.json')
    parser.add_argument('--val_annotations', type=str, default='data/annotations/captions_val2014.json')

    # karpathy's splits
    parser.add_argument('--train_split', type=str, default='data/splits/coco_train.txt')
    parser.add_argument('--test_split', type=str, default='data/splits/coco_test.txt')
    parser.add_argument('--val_split', type=str, default='data/splits/coco_val.txt')
    parser.add_argument('--restval_split', type=str, default='data/splits/coco_restval.txt')

    # official splits
    parser.add_argument('--train_split_online', type=str, default='data/splits/coco_train_online.txt')
    parser.add_argument('--val_split_online', type=str, default='data/splits/coco_val_online.txt')

    # params of preprocessed data
    parser.add_argument('--train_val_imageNames_to_imageIDs_path', type=str, default='data/train_val_imageNames_to_imageIDs.pkl')
    parser.add_argument('--official_train_captions_path', type=str, default='data/train_images_captions_official.pkl')
    parser.add_argument('--official_val_captions_path', type=str, default='data/val_images_captions_official.pkl')
    parser.add_argument('--train_images_captions_path', type=str, default='data/train_images_captions.pkl')
    parser.add_argument('--val_images_captions_path', type=str, default='data/val_images_captions.pkl')
    parser.add_argument('--word_to_idx_path', type=str, default='data/word_to_idx.pkl')
    parser.add_argument('--idx_to_word_path', type=str, default='data/idx_to_word.pkl')
    parser.add_argument('--bias_init_vector_path', type=str, default='data/bias_init_vector.pkl')
    parser.add_argument('--train_images_captions_index', type=str, default='data/train_images_captions_index.pkl')

    # params of focal loss
    parser.add_argument('--focal_gamma', type=float, default=2)

    # batch normalization
    parser.add_argument('--epsilon', type=float, default=0.001)

    # label smoothing
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # params of t-SNE visualization
    parser.add_argument('--vis_batch_size', type=int, default=64)
    parser.add_argument('--vis_model_path', type=str, default='')
    parser.add_argument('--vis_save_path', type=str, default='')

    # ARNet
    parser.add_argument('--rcst_time', type=int, default=1)
    parser.add_argument('--rcst_size', type=int, default=512)
    parser.add_argument('--rcst_weight', type=float, default=0.005)
    parser.add_argument('--rcst_learning_rate', type=float, default=0.0005)
    parser.add_argument('--rcst_model_save_path', type=str, default='')
    parser.add_argument('--rcst_model_path', type=str, default='')
    parser.add_argument('--rcst_train_json_path', type=str, default='')

    # params of ZoneOut
    parser.add_argument('--zoneout_factor_cell', type=float, default=0.1)
    parser.add_argument('--zoneout_factor_output', type=float, default=0.0)

    # ensemble model
    parser.add_argument('--ensemble_file_path', type=str, default='data/splits/coco_test.txt')
    parser.add_argument('--ensemble_json_path', type=str, default='')
    parser.add_argument('--ensemble_model_0', type=str, default='')
    parser.add_argument('--ensemble_model_1', type=str, default='')
    parser.add_argument('--ensemble_model_2', type=str, default='')
    parser.add_argument('--ensemble_model_3', type=str, default='')
    parser.add_argument('--ensemble_model_4', type=str, default='')
    parser.add_argument('--ensemble_model_5', type=str, default='')

    # inference function parameters
    parser.add_argument('--infer_file_path', type=str, default='data/splits/coco_test.txt')
    parser.add_argument('--infer_json_path', type=str, default='')
    parser.add_argument('--infer_model_path', type=str, default='')

    # self-critical training function parameters
    parser.add_argument('--grad_clip', type=float, default=0.1,
                        help='clip gradients at this value')

    parser.add_argument('--scst_learning_rate', type=float, default=0.00005)
    parser.add_argument('--scst_learning_rate_decay', type=float, default=0.8)
    parser.add_argument('--scst_loss_mixed_gamma', type=float, default=0.0016)
    parser.add_argument('--scst_batch_size', type=int, default=8)
    parser.add_argument('--scst_epochs', type=int, default=200)
    parser.add_argument('--reward_method', type=str, default='CIDEr')
    parser.add_argument('--scst_base_model_path', type=str, default='')
    parser.add_argument('--scst_model_save_path', type=str, default='')
    parser.add_argument('--scst_train_json_path', type=str, default='')

    # caption_with_beam_search function parameters
    parser.add_argument('--beam_model_path', type=str, default='')
    parser.add_argument('--beam_file_path', type=str, default='data/splits/coco_test.txt')
    parser.add_argument('--beam_json_path', type=str, default='')
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--beam_length_normalization_factor', type=float, default=0.0)

    # from Ruotian Luo
    parser.add_argument('--train_only', type=int, default=0)
    parser.add_argument('--caption_model', type=str, default='caption_model')
    parser.add_argument('--input_json', type=str, default='data/cocotalk.json')
    parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5')
    parser.add_argument('--load_best_score', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)

    # XE 训练的一些参数, 路径
    parser.add_argument('--train_json_path', type=str, default='')
    parser.add_argument('--model_save_basepath', type=str, default='models')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--xe_model_save_path', type=str, default='')

    # 从上次训练保存的 model 文件开始接着训练
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--start_from_epoch', type=int, default=0)

    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=110)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seq_per_img', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--early_stop_value', type=int, default=10)

    # params of optimizer
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--learning_rate_decay_start', type=int, default=0)
    parser.add_argument('--learning_rate_decay_every', type=int, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,)
    parser.add_argument('--optim_alpha', type=float, default=0.9)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--optim_weight_decay', type=float, default=0.00001)

    # params of ZoneOut
    parser.add_argument('--c_ratio', type=float, default=0.0)
    parser.add_argument('--h_ratio', type=float, default=0.0)

    # params of Scheduled Sampling (SS)
    parser.add_argument('--ss_prob', type=float, default=0.0)
    parser.add_argument('--ss_start', type=int, default=-1)
    parser.add_argument('--ss_increase_every', type=int, default=5)
    parser.add_argument('--ss_increase_prob', type=float, default=0.05)
    parser.add_argument('--ss_max_prob', type=float, default=0.25)

    # params of ReviewNet
    parser.add_argument('--n_reviewers', type=int, default=8)

    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--lstm_size', type=int, default=512)
    parser.add_argument('--lstm_step', type=int, default=30)
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--vocab_size', type=int, default=10516)
    parser.add_argument('--word_embed_size', type=int, default=512)
    parser.add_argument('--conv_feat_size', type=int, default=1536)
    parser.add_argument('--conv_att_size', type=int, default=64)
    parser.add_argument('--fc_feat_size', type=int, default=1536)
    parser.add_argument('--att_hidden_size', type=int, default=512)
    parser.add_argument('--top_word_count', type=int, default=1000)
    parser.add_argument('--feature_type', type=str, default='inception_v4')

    # conv features and fc features, default is inception-v4
    parser.add_argument('--conv_feat_path', type=str, default='data/feats/mscoco_feats_v4_conv')
    parser.add_argument('--fc_feat_path', type=str, default='data/feats/mscoco_feats_v4_fc')

    args = parser.parse_args()

    if args.feature_type == 'inception_v3':
        args.conv_feat_path = 'data/feats/train_val_test_feats_v3_conv'
        args.fc_feat_path = 'data/feats/train_val_test_feats_v3_fc'

        args.conv_feat_size = 1280
        args.conv_att_size = 64
        args.fc_feat_size = 2048

    elif args.feature_type == 'inception_v4':
        args.conv_feat_path = 'data/feats/mscoco_feats_v4_conv'
        args.fc_feat_path = 'data/feats/mscoco_feats_v4_fc'

        args.conv_feat_size = 1536
        args.conv_att_size = 64
        args.fc_feat_size = 1536

    elif args.feature_type == 'densenet':
        args.input_fc_dir = 'data/feats/train_val_test_feats_densenet161_conv'
        args.input_att_dir = 'data/feats/train_val_test_feats_densenet161_conv'

        args.conv_feat_size = 2208
        args.conv_att_size = 49
        args.fc_feat_size = 2208

    return args


if __name__ == '__main__':
    opt = parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))
