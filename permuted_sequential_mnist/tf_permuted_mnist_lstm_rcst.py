#! encoding: UTF-8

import os
import opts
import ipdb
import time
import random
import numpy as np
from six.moves import cPickle

import keras
from keras.datasets import mnist
import tensorflow as tf


class LSTM():
    def __init__(self, opt):
        self.lstm_step = opt.lstm_step
        self.batch_size = opt.batch_size
        self.lstm_size = opt.lstm_size

        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
        self.logit_W = tf.Variable(tf.random_uniform([self.lstm_size, 10], -0.08, 0.08), name='logit_W')
        self.logit_b = tf.Variable(tf.zeros([10]), name='logit_b')

        # ARNet
        self.rcst_weight = opt.rcst_weight
        self.lstm_rcst = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
        self.linear_rcst_W = tf.Variable(tf.random_uniform([self.lstm_size, self.lstm_size], -0.08, 0.08), name='linear_rcst_W')
        self.linear_rcst_b = tf.Variable(tf.zeros([self.lstm_size]), name='linear_rcst_b')

    def build_model(self):
        pixels = tf.placeholder(tf.float32, [self.batch_size, self.lstm_step, 1])
        onehot_labels = tf.placeholder(tf.int32, [self.batch_size, 10])

        state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        state_rcst = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        prev_h = state[1]
        rcst_loss = 0.0
        for i in range(0, self.lstm_step):
            with tf.variable_scope("LSTM"):
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()
                output, state = self.lstm(pixels[:, i], state)

            if i == self.lstm_step - 1:
                logit_outputs = tf.matmul(output, self.logit_W) + self.logit_b
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_outputs)
                
                loss = tf.reduce_sum(cross_entropy) / self.batch_size

            with tf.variable_scope("rcst"):
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()
                output_rcst, state_rcst = self.lstm_rcst(output, state_rcst)

            output_rcst_proj = tf.matmul(output_rcst, self.linear_rcst_W) + self.linear_rcst_b
            current_rcst_loss = tf.reduce_sum(tf.square((output_rcst_proj - prev_h))) * self.rcst_weight
            current_rcst_loss /= self.batch_size
            rcst_loss += current_rcst_loss

        return loss, rcst_loss, logit_outputs, pixels, onehot_labels


def train(opt, x_train, y_train, x_test, y_test):
    tf.set_random_seed(opt.seed)

    model = LSTM(opt)
    tf_loss, tf_rcst_loss, tf_logit_outputs, tf_pixels, tf_onehot_labels = model.build_model()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # restore from the pre-trained model
    saver = tf.train.Saver(max_to_keep=opt.max_epochs, write_version=1)
    saver.restore(sess, opt.rcst_model_base_path)

    tf_learning_rate = tf.placeholder(tf.float32)
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        train_op = tf.train.AdamOptimizer(tf_learning_rate).minimize(tf_loss)

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    if len(uninitialized_vars):
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)

    iter_cnt = 0

    for epoch in range(0, opt.max_epochs):
        if epoch == 0:
            current_learning_rate = opt.learning_rate
        elif epoch != 0 and epoch % opt.learning_rate_decay_every == 0:
            current_learning_rate = current_learning_rate * opt.learning_rate_decay_rate

        # training
        for start, end in zip(range(0, x_train.shape[0], opt.batch_size),
                              range(opt.batch_size, x_train.shape[0], opt.batch_size)):
            start_time = time.time()

            current_batch_pixels_train = x_train[start:end]
            current_batch_labels_train = y_train[start:end]

            feed_dict = {tf_learning_rate: current_learning_rate,
                         tf_pixels: current_batch_pixels_train,
                         tf_onehot_labels: current_batch_labels_train}

            _, loss_val, rcst_loss_val, logit_outputs = sess.run([train_op, tf_loss, tf_rcst_loss, tf_logit_outputs], feed_dict)

            # 计算训练精度
            pred_y = np.argmax(logit_outputs, axis=1)
            true_y = np.argmax(current_batch_labels_train, axis=1)
            current_acc = sum(pred_y == true_y) / float(opt.batch_size)

            iter_cnt += 1
            end_time = time.time()
            print("iter {:4d}  epoch {:3d}  lr {:.5f}  loss {:.4f}  rcst_loss {:.4f}  train_acc {:.4f}  time batch {:.4f}".format(iter_cnt,
                epoch, current_learning_rate, loss_val, rcst_loss_val, current_acc, end_time-start_time))

        # validation
        if np.mod(epoch, 1) == 0:
            print("epoch {} is done, saving the model ...".format(epoch))
            saver.save(sess, os.path.join(opt.rcst_model_save_path, 'model_epoch'), global_step=epoch)

            true_cnt = 0
            test_batch_cnt = 0
            for start, end in zip(range(0, x_test.shape[0], opt.batch_size),
                                  range(opt.batch_size, x_test.shape[0], opt.batch_size)):

                current_batch_pixels_test = x_test[start:end]
                current_batch_labels_test = y_test[start:end]

                feed_dict = {tf_learning_rate: current_learning_rate,
                             tf_pixels: current_batch_pixels_test,
                             tf_onehot_labels: current_batch_labels_test}

                loss_test, logit_outputs = sess.run([tf_loss, tf_logit_outputs], feed_dict)

                # 计算验证精度
                pred_y = np.argmax(logit_outputs, axis=1)
                true_y = np.argmax(current_batch_labels_test, axis=1)
                true_cnt += sum(pred_y == true_y)
                test_batch_cnt += 1

            test_acc = true_cnt / float(test_batch_cnt * opt.batch_size)
            print("epoch {}  test_acc {:.4f}  test_num: {}".format(epoch, test_acc, test_batch_cnt * opt.batch_size))


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))

    with open('permuted_mnist_110.pkl', 'rb') as f:
        permuted_mnist = cPickle.load(f)

    if os.path.isdir(opt.rcst_model_save_path) is False:
        os.mkdir(opt.rcst_model_save_path)

    x_train_permuted = permuted_mnist['x_train_permuted']
    y_train = permuted_mnist['y_train']

    x_test_permuted = permuted_mnist['x_test_permuted']
    y_test = permuted_mnist['y_test']

    # training
    train(opt, x_train_permuted, y_train, x_test_permuted, y_test)

