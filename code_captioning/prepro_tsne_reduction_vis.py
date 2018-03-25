from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

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
    """ 
    compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    """
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return 1.0 - sumxy / math.sqrt(sumxx * sumyy)


def dim_reduction(opt):
    teacher_forcing_hidden_reduction = []
    free_running_hidden_reduction = []

    for start, end in [(opt.vis_batch_size * opt.truncation, opt.vis_batch_size * (opt.truncation + 1))]:
        current_teacher_forcing_hidden = teacher_forcing_hidden[start:end, :]
        current_free_running_hidden = free_running_hidden[start:end, :]
        current_hidden = np.concatenate((current_teacher_forcing_hidden, current_free_running_hidden), axis=0)

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)        
        current_hidden_reduction = tsne.fit_transform(current_hidden)

        teacher_forcing_hidden_reduction.append(current_hidden_reduction[0:opt.vis_batch_size, :])
        free_running_hidden_reduction.append(current_hidden_reduction[opt.vis_batch_size:, :])

    teacher_forcing_hidden_reduction = np.reshape(teacher_forcing_hidden_reduction, [-1, 2])
    free_running_hidden_reduction = np.reshape(free_running_hidden_reduction, [-1, 2])
    hidden_reduction = np.concatenate((teacher_forcing_hidden_reduction, free_running_hidden_reduction), axis=0)

    return hidden_reduction


def calculate_distance(teacher_forcing_hidden, free_running_hidden):
    bsize = free_running_hidden.shape[0]
    mean_teacher_forcing = np.mean(teacher_forcing_hidden, axis=0)
    mean_free_running = np.mean(free_running_hidden, axis=0)

    distance_mc = np.sqrt(np.sum(np.square(mean_teacher_forcing - mean_free_running)))
    distance_pw = np.sqrt(np.sum(np.square(teacher_forcing_hidden - free_running_hidden), 1))

    print("distance_mc: {}".format(distance_mc))
    print("distance_pw: {}".format(np.sum(distance_pw) / bsize))

    # cosine distance
    distance_cosine = 0.0
    for i in range(bsize):
        current_tf_h = teacher_forcing_hidden[i]
        current_fr_h = free_running_hidden[i]
        distance_cosine += cosine_similarity(current_tf_h, current_fr_h)

    print("cosine distance_pw: {}".format(distance_cosine / bsize))
    print("cosine distance_mc: {}".format(cosine_similarity(mean_teacher_forcing, mean_free_running)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hidden_path', type=str, 
        default='example: models/soft_attention_seed_117/model_epoch-51_hidden_states.pkl')
    
    parser.add_argument('--hidden_reduction_save_path', type=str, 
        default='example: models/soft_attention_seed_117/model_epoch-51_hidden_states_reduction.pkl')
    
    parser.add_argument('--vis_batch_size', type=int, default=80)
    parser.add_argument('--truncation', type=int, default=20)
    opt = parser.parse_args()

    with open(opt.hidden_path, 'rb') as f:
        hidden_states = cPickle.load(f)

    teacher_forcing_hidden = np.squeeze(hidden_states['teacher_forcing'])
    free_running_hidden = np.squeeze(hidden_states['free_running'])

    # calculate the distances
    calculate_distance(teacher_forcing_hidden, free_running_hidden)

    # dimensionality reduction
    hidden_reduction = dim_reduction(opt)
    
    with open(opt.hidden_reduction_save_path, 'wb') as f:
        cPickle.dump(hidden_reduction, f)

    # visualize the hidden states after dimensionality reduction
    plot_embedding(hidden_reduction, opt.vis_batch_size, "t-SNE visilization")
    plt.show()

