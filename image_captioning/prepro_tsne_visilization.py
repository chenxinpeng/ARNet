#! encoding: UTF-8

import os
import ipdb
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import opts

from six.moves import cPickle
from sklearn import manifold


def plot_embedding(X, batch_size, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()

    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    for i in range(X.shape[0]):
        if i < batch_size:
            ax.plot(X[i, 0], X[i, 1], color='yellowgreen', marker='.')  # teacher forcing hidden states
        else:
            ax.plot(X[i, 0], X[i, 1], color='orangered', marker='.')  # free running hidden states


def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return 1.0 - sumxy / math.sqrt(sumxx * sumyy)


if __name__ == "__main__":
    opt = opts.parse_opt()

    if opt.version == 'offline':
        train_images_names = open(opt.train_split, 'r').read().splitlines() + open(opt.restval_split, 'r').read().splitlines()
        val_images_names = open(opt.val_split, 'r').read().splitlines()
    elif opt.version == 'online':
        train_images_names = open(opt.train_split_online, 'r').read().splitlines()
        val_images_names = open(opt.val_split_online, 'r').read().splitlines()

    val_images_names = sorted(val_images_names)

    # 正常 Soft attention 得到的 hidden states
    hidden_states_path = 'models/soft_attention_inception_v4_seed_117/model_epoch-8_hidden_states.pkl'
    hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_117/model_epoch-8_hidden_states_reduction.pkl'

    # reconstruct h_{t-1} 得到的 hidden states, 6
    #hidden_states_path = 'models/soft_attention_inception_v4_seed_117_reconstruct_2nd_0.009_2nd/model_epoch-0_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_117_reconstruct_2nd_0.009_2nd/model_epoch-0_hidden_states_reduction.pkl'

    # ss 得到的 hidden states
    #hidden_states_path = 'models/soft_attention_inception_v4_seed_112_ss/model_epoch-12_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_112_ss/model_epoch-12_hidden_states_reduction.pkl'

    # reconstruct h_{t-1}, context 得到的 hidden states
    #hidden_states_path = 'models/soft_attention_inception_v4_seed_110_reconstruct_0.005/model_epoch-6_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_110_reconstruct_0.005/model_epoch-6_hidden_states_reduction.pkl'

    # zoneout 得到的 hidden states
    #hidden_states_path = 'models/soft_attention_inception_v4_seed_110_zoneout_c_0.05_h_0.05/model_epoch-9_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_110_zoneout_c_0.05_h_0.05/model_epoch-9_hidden_states_reduction.pkl'

    # SCST 得到的 hidden states
    #hidden_states_path = 'models/soft_attention_inception_v4_seed_110_scst/model_epoch-29_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_110_scst/model_epoch-29_hidden_states_reduction.pkl'

    #seed = 117
    #index = 20
    #hidden_states_path = 'models/encoder_decoder_ss_inception_v4_seed_' + str(seed) + '/model_epoch-' + str(index) + '_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/encoder_decoder_ss_inception_v4_seed_' + str(seed) + '/model_epoch-' + str(index) + '_hidden_states_reduction.pkl'

    #hidden_states_path = 'models/encoder_decoder_inception_v4_seed_110_reconstruct_0.1/model_epoch-64_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/encoder_decoder_inception_v4_seed_110_reconstruct_0.1/model_epoch-64_hidden_states_reduction.pkl'

    #hidden_states_path = 'models/encoder_decoder_zoneout_inception_v4_seed_110_zoneout_c_0.05_h_0.05/model_epoch-31_hidden_states.pkl'
    #hidden_states_reduction_path = 'models/encoder_decoder_zoneout_inception_v4_seed_110_zoneout_c_0.05_h_0.05/model_epoch-31_hidden_states_reduction.pkl'

    with open(hidden_states_path, 'rb') as f:
        hidden_states = cPickle.load(f)

    teacher_forcing_hidden_states = np.squeeze(hidden_states['teacher_forcing'])
    free_running_hidden_states = np.squeeze(hidden_states['free_running'])

    shape_0 = free_running_hidden_states.shape[0]
    print(shape_0)
    mean_teacher_forcing = np.mean(teacher_forcing_hidden_states, axis=0)
    mean_free_running = np.mean(free_running_hidden_states, axis=0)

    distance = np.sqrt(np.sum(np.square(mean_teacher_forcing - mean_free_running)))
    distance_2 = np.sqrt(np.sum(np.square(teacher_forcing_hidden_states - free_running_hidden_states), 1))

    print("distance_mc: {}".format(distance))
    print("distance_pw: {}".format(np.sum(distance_2)/shape_0))

    # cosine distance
    distance_cosine = 0.0
    for i in range(shape_0):
        current_tf_h = teacher_forcing_hidden_states[i]
        current_fr_h = free_running_hidden_states[i]
        distance_cosine += cosine_similarity(current_tf_h, current_fr_h)

    print("cosine distance_pw: {}".format(distance_cosine/shape_0))
    print("cosine distance_mc: {}".format(cosine_similarity(mean_teacher_forcing, mean_free_running)))

    teacher_forcing_hidden_states_reduction = []
    free_running_hidden_states_reduction = []

    opt.t_SNE_batch_size = 80
    len_images = teacher_forcing_hidden_states.shape[0]

    # 20
    fac = 5

    # for start, end in zip(range(0, len_images, opt.t_SNE_batch_size), range(opt.t_SNE_batch_size, len_images, opt.t_SNE_batch_size)):
    for start, end in [(opt.t_SNE_batch_size * fac, opt.t_SNE_batch_size * (fac + 1))]:
        print("{}  {}".format(start, end))

        current_teacher_forcing_hidden_states = teacher_forcing_hidden_states[start:end, :]
        current_free_running_hidden_states = free_running_hidden_states[start:end, :]
        current_hidden_states = np.concatenate((current_teacher_forcing_hidden_states, current_free_running_hidden_states), axis=0)

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        #tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, metric="cosine")
        current_hidden_states_reduction = tsne.fit_transform(current_hidden_states)

        teacher_forcing_hidden_states_reduction.append(current_hidden_states_reduction[0:opt.t_SNE_batch_size, :])
        free_running_hidden_states_reduction.append(current_hidden_states_reduction[opt.t_SNE_batch_size:, :])

    teacher_forcing_hidden_states_reduction = np.reshape(teacher_forcing_hidden_states_reduction, [-1, 2])
    free_running_hidden_states_reduction = np.reshape(free_running_hidden_states_reduction, [-1, 2])
    hidden_states_reduction = np.concatenate((teacher_forcing_hidden_states_reduction, free_running_hidden_states_reduction), axis=0)

    with open(hidden_states_reduction_path, 'wb') as f:
        cPickle.dump(hidden_states_reduction, f)

    plot_embedding(hidden_states_reduction, opt.t_SNE_batch_size, "t-SNE visilization")
    plt.show()

