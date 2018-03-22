import tensorflow as tf

seed = 110

OLD_CHECKPOINT_FILE = 'models/permuted_mnist_lstm_seed_110/model_epoch-257'
NEW_CHECKPOINT_FILE = 'models/permuted_mnist_lstm_seed_110/model_epoch-257_rcst'

new_checkpoint_vars = {}

reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)

new_checkpoint_vars['linear_rcst_W'] = tf.Variable(tf.random_uniform([128, 128], -0.08, 0.08), name='linear_rcst_W')
new_checkpoint_vars['linear_rcst_b'] = tf.Variable(tf.zeros([128]), name='linear_rcst_b')

new_checkpoint_vars['rcst/basic_lstm_cell/weights'] = tf.Variable(tf.random_uniform([256, 512], -0.08, -0.08))
new_checkpoint_vars['rcst/basic_lstm_cell/biases'] = tf.Variable(tf.zeros(512))

for old_name in reader.get_variable_to_shape_map():
    new_checkpoint_vars[old_name] = tf.Variable(reader.get_tensor(old_name))

tf.set_random_seed(seed)
init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars, write_version=1)

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, NEW_CHECKPOINT_FILE)