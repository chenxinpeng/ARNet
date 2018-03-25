import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file_path', type=str, default='data/train.dat')
    parser.add_argument('--val_file_path', type=str, default='data/dev.dat')
    parser.add_argument('--test_file_path', type=str, default='data/test.dat')

    parser.add_argument('--prefix_path', type=str, default='data/prefix.pkl')
    parser.add_argument('--train_data_path', type=str, default='data/train_data.pkl')
    parser.add_argument('--val_data_path', type=str, default='data/val_data.pkl')
    parser.add_argument('--test_data_path', type=str, default='data/test_data.pkl')

    parser.add_argument('--token2index_path', type=str, default='data/token2index.pkl')
    parser.add_argument('--index2token_path', type=str, default='data/index2token.pkl')

    parser.add_argument('--function', type=str, default='train')
    parser.add_argument('--model_save_path', type=str, default='')
    parser.add_argument('--test_model_path', type=str, default='')

    # params of optimizer
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--learning_rate_decay_every', type=int, default=2)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--optim_alpha', type=float, default=0.9)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--optim_weight_decay', type=float, default=0.00001)

    parser.add_argument('--word_cnt', type=int, default=12859)
    parser.add_argument('--token_cnt', type=int, default=20396)
    parser.add_argument('--vocab_size', type=int, default=12860)
    parser.add_argument('--code_truncate', type=int, default=300)
    parser.add_argument('--comment_truncate', type=int, default=300)

    parser.add_argument('--seed', type=int, default=110)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lstm_size', type=int, default=256)
    parser.add_argument('--att_hidden_size', type=int, default=512)
    parser.add_argument('--encoding_att_size', type=int, default=300)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--input_encoding_size', type=int, default=50)

    # params for reviewnet
    parser.add_argument('--num_review_steps', type=int, default=8)
    parser.add_argument('--drop_prob_reason', type=float, default=0.1)

    parser.add_argument('--rcst_weight', type=float, default=0.005)
    parser.add_argument('--rcst_model_base_path', type=str, default='')
    parser.add_argument('--rcst_model_save_path', type=str, default='')

    # params of Scheduled Sampling (SS)
    parser.add_argument('--is_ss', type=bool, default=False)
    parser.add_argument('--ss_prob', type=float, default=0.0)
    parser.add_argument('--ss_start', type=int, default=-1)
    parser.add_argument('--ss_increase_every', type=int, default=5)
    parser.add_argument('--ss_increase_prob', type=float, default=0.05)
    parser.add_argument('--ss_max_prob', type=float, default=0.25)

    parser.add_argument('--vis_batch_size', type=int, default=16)
    parser.add_argument('--vis_model_path', type=str, default='')
    parser.add_argument('--vis_save_path', type=str, default='')

    # params of ZoneOut
    parser.add_argument('--c_ratio', type=float, default=0.0)
    parser.add_argument('--h_ratio', type=float, default=0.0)

    parser.add_argument('--grads_save_path', type=str, default='')
    parser.add_argument('--sentence_save_path', type=str, default='')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    opt = parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))
