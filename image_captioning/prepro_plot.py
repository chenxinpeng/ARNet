import os
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle


def plot_embedding(X1, X2, X3, batch_size, title=None):
    x_min, x_max = np.min(X1, 0), np.max(X1, 0)
    X1 = (X1 - x_min) / (x_max - x_min)
    center_teacher_forcing = np.mean(X1[:batch, :], axis=0, keepdims=False)
    center_free_running = np.mean(X1[batch:, :], axis=0, keepdims=False)

    x_min, x_max = np.min(X2, 0), np.max(X2, 0)
    X2 = (X2 - x_min) / (x_max - x_min)
    reconstruct_center_teacher_forcing = np.mean(X2[:batch, :], axis=0, keepdims=False)
    reconstruct_center_free_running = np.mean(X2[batch:, :], axis=0, keepdims=False)

    x_min, x_max = np.min(X3, 0), np.max(X3, 0)
    X3 = (X3 - x_min) / (x_max - x_min)
    ss_center_teacher_forcing = np.mean(X2[:batch, :], axis=0, keepdims=False)
    ss_center_free_running = np.mean(X2[batch:, :], axis=0, keepdims=False)

    fig = plt.figure(frameon=False)
    fig.suptitle('', fontsize=14, fontweight='bold')

    ax_1 = fig.add_subplot(231, aspect='equal')
    ax_2 = fig.add_subplot(232, aspect='equal')
    ax_3 = fig.add_subplot(233, aspect='equal')
    #ax_4 = fig.add_subplot(234, aspect='equal')
    #ax_5 = fig.add_subplot(235, aspect='equal')
    #ax_6 = fig.add_subplot(236, aspect='equal')

    ax_1.get_xaxis().set_visible(False)
    ax_1.get_yaxis().set_visible(False)

    ax_2.get_xaxis().set_visible(False)
    ax_2.get_yaxis().set_visible(False)

    ax_3.get_xaxis().set_visible(False)
    ax_3.get_yaxis().set_visible(False)

    #ax_4.get_xaxis().set_visible(False)
    #ax_4.get_yaxis().set_visible(False)

    #ax_5.get_xaxis().set_visible(False)
    #ax_5.get_yaxis().set_visible(False)

    #ax_6.get_xaxis().set_visible(False)
    #ax_6.get_yaxis().set_visible(False)

    ax_1.axis([0.0, 1.0, 0.0, 1.0])
    ax_2.axis([0.0, 1.0, 0.0, 1.0])
    ax_3.axis([0.0, 1.0, 0.0, 1.0])
    #ax_4.axis([0.0, 1.0, 0.0, 1.0])
    #ax_5.axis([0.0, 1.0, 0.0, 1.0])
    #ax_6.axis([0.0, 1.0, 0.0, 1.0])

    #ax_4.plot(center_teacher_forcing[0], center_teacher_forcing[1], color='deepskyblue', marker='x')
    #ax_4.plot(center_free_running[0], center_free_running[1], color='orangered', marker='x')

    #ax_5.plot(reconstruct_center_teacher_forcing[0], reconstruct_center_teacher_forcing[1], color='deepskyblue', marker='x')
    #ax_5.plot(reconstruct_center_free_running[0], reconstruct_center_free_running[1], color='orangered', marker='x')

    #ax_6.plot(ss_center_teacher_forcing[0], ss_center_teacher_forcing[1], color='deepskyblue', marker='x')
    #ax_6.plot(ss_center_free_running[0], ss_center_free_running[1], color='orangered', marker='x')

    for i in range(X1.shape[0]):
        if i < batch_size:
            ax_1.plot(X1[i, 0], X1[i, 1], color=[(1.0, 0.701, 0.701)], marker='.')  # teacher forcing
            ax_2.plot(X2[i, 0], X2[i, 1], color=[(1.0, 0.701, 0.701)], marker='.')
            ax_3.plot(X3[i, 0], X3[i, 1], color=[(1.0, 0.701, 0.701)], marker='.')
        else:
            ax_1.plot(X1[i, 0], X1[i, 1], color=[(0.701, 0.701, 1.0)], marker='.')  # free running
            ax_2.plot(X2[i, 0], X2[i, 1], color=[(0.701, 0.701, 1.0)], marker='.')
            ax_3.plot(X3[i, 0], X3[i, 1], color=[(0.701, 0.701, 1.0)], marker='.')

    # plt.axis([0.0, 1, 0.0, 1.0])
    plt.show()


def plot_single(X, batch_size):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    center_teacher_forcing = np.mean(X[:batch, :], axis=0, keepdims=False)
    center_free_running = np.mean(X[batch:, :], axis=0, keepdims=False)

    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111, aspect='equal')

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # normal
    #ax.axis([0.0, 0.8, 0.1, 0.9])

    # ss
    #ax.axis([0.25, 0.85, 0.15, 0.75])

    # reconstruct
    #ax.axis([-0.3, 1.3, -0.3, 1.3])

    # zoneout
    #ax.axis([-0.2, 1.2, -0.2, 1.2])

    ax.axis([-0.2, 1.2, -0.2, 1.2])

    for i in range(X.shape[0]):
        if i < batch_size:
            ax.plot(X[i, 0], X[i, 1],
                    color="dodgerblue",
                    marker='.',
                    markersize=11)  # teacher forcing
        else:
            ax.plot(X[i, 0], X[i, 1],
                    markeredgecolor="orangered",
                    markeredgewidth=1,
                    markerfacecolor='none',
                    marker='.',
                    markersize=11)  # free running

    # plt.axis([0.0, 1, 0.0, 1.0])
    # plt.show()


if __name__ == "__main__":
    hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_117/model_epoch-8_hidden_states_reduction.pkl'

    reconstruct_hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_117_reconstruct_2nd_0.009_2nd/model_epoch-0_hidden_states_reduction.pkl'

    #reconstruct_hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_112_reconstruct_2nd_0.1/model_epoch-13_hidden_states_reduction.pkl'

    ss_hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_112_ss/model_epoch-12_hidden_states_reduction.pkl'

    zoneout_hidden_states_reduction_path = 'models/soft_attention_inception_v4_seed_110_zoneout_c_0.05_h_0.05/model_epoch-9_hidden_states_reduction.pkl'

    with open(hidden_states_reduction_path, 'rb') as f:
        hidden_states_reduction = pickle.load(f)

    with open(reconstruct_hidden_states_reduction_path, 'rb') as f:
        reconstruct_hidden_states_reduction = pickle.load(f)

    with open(ss_hidden_states_reduction_path, 'rb') as f:
        ss_hidden_states_reduction = pickle.load(f)

    with open(zoneout_hidden_states_reduction_path, 'rb') as f:
        zoneout_hidden_states_reduction = pickle.load(f)

    batch = int(hidden_states_reduction.shape[0] / 2)

    # plot_embedding(hidden_states_reduction, ss_hidden_states_reduction, reconstruct_hidden_states_reduction, batch, title='t-SNE visilization')

    #plot_single(hidden_states_reduction, batch)
    #plt.savefig("/Users/chenxinpeng/Dropbox/aaai_writing/figures/visualization_hidden_states_reduction.pdf", format='pdf', dpi=1000, bbox_inches='tight')

    plot_single(reconstruct_hidden_states_reduction, batch)
    plt.savefig("/Users/chenxinpeng/Dropbox/aaai_writing/figures/visualization_reconstruct_hidden_states_reduction.pdf", format='pdf', dpi=1000, bbox_inches='tight')

    #plot_single(ss_hidden_states_reduction, batch)
    #plt.savefig("/Users/chenxinpeng/Dropbox/aaai_writing/figures/visualization_ss_hidden_states_reduction.pdf", format='pdf', dpi=1000, bbox_inches='tight')

    #plot_single(zoneout_hidden_states_reduction, batch)
    #plt.savefig("/Users/chenxinpeng/Dropbox/aaai_writing/figures/visualization_zoneout_hidden_states_reduction.pdf", format='pdf', dpi=1000, bbox_inches='tight')
