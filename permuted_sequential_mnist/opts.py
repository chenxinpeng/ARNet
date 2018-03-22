#! encoding: UTF-8
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_cuda', type=bool, default=False)

    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--download_mnist', type=bool, default=False)
    parser.add_argument('--mnist_path', type=str, default='mnist.npz')

    parser.add_argument('--function', type=str, default='train')
    parser.add_argument('--model_save_path', type=str, default='models/mnist_lstm_seed_110')
    parser.add_argument('--test_model_path', type=str, default='models/mnist_lstm_seed_110')

    # 优化器, 学习率参数
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_decay_every', type=int, default=2)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--optim_alpha', type=float, default=0.9)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--optim_weight_decay', type=float, default=0.00001)

    parser.add_argument('--seed', type=int, default=110)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lstm_size', type=int, default=128)
    parser.add_argument('--lstm_step', type=int, default=784)
    parser.add_argument('--input_size', type=int, default=28)
    parser.add_argument('--drop_prob', type=float, default=0.0)

    parser.add_argument('--rcst_weight', type=float, default=0.005)
    parser.add_argument('--rcst_model_base_path', type=str, default='')
    parser.add_argument('--rcst_model_save_path', type=str, default='')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    opt = parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))
