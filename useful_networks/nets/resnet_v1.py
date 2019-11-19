import math
import tensorflow as tf
import numpy as np
from tensorflow.python import control_flow_ops
slim = tf.contrib.slim


def conv2d(x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
  with tf.variable_scope(scope):
    kernel = tf.Variable(
      tf.truncated_normal([k, k, n_in, n_out],
        stddev=math.sqrt(2/(k*k*n_in))),
      name='weight')
    tf.add_to_collection('weights', kernel)
    conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
    if bias:
      bias = tf.get_variable('bias', [n_out], initializer=tf.constant_initializer(0.0))
      tf.add_to_collection('biases', bias)
      conv = tf.nn.bias_add(conv, bias)
  return conv


def batch_norm(x, phase_train, scope, epsilon=1e-3, decay=0.99, reuse=tf.AUTO_REUSE):
    """ Assume nd [batch, N1, N2, ..., Nm, Channel] tensor"""
    with tf.variable_scope(scope, reuse=reuse):
        size = x.get_shape().as_list()[-1]
        scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size])

        pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, list(range(len(x.get_shape())-1)))
        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(phase_train, batch_statistics, population_statistics)


def residual_block(x, n_in, n_out, subsample, phase_train, scope='res_block', reuse=tf.AUTO_REUSE):
  with tf.variable_scope(scope):
    if subsample:
      y = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='conv_1')
      shortcut = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='shortcut')
    else:
      y = conv2d(x, n_in, n_out, 3, 1, 'SAME', False, scope='conv_1')
      shortcut = tf.identity(x, name='shortcut')
    y = batch_norm(y, phase_train, scope='bn_1', reuse=reuse)
    y = tf.nn.relu(y, name='relu_1')
    y = conv2d(y, n_out, n_out, 3, 1, 'SAME', True, scope='conv_2')
    y = batch_norm(y, phase_train, scope='bn_2', reuse=reuse)
    y = y + shortcut
    y = tf.nn.relu(y, name='relu_2')
  return y


def residual_group(x, n_in, n_out, n, first_subsample, phase_train, scope='res_group', reuse=tf.AUTO_REUSE):
  with tf.variable_scope(scope):
    y = residual_block(x, n_in, n_out, first_subsample, phase_train, scope='block_1', reuse=reuse)
    for i in range(n - 1):
      y = residual_block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i + 2), reuse=reuse)
  return y


def residual_net(x, n, n_classes, phase_train, scope='res_net', reuse=tf.AUTO_REUSE):
    end_point = {}
    with tf.variable_scope(scope):
        y = conv2d(x, 3, 16, 3, 1, 'SAME', False, scope='conv_init')
        y = batch_norm(y, phase_train, scope='bn_init', reuse=reuse)
        y = tf.nn.relu(y, name='relu_init')
        y = end_point['group_1'] = residual_group(y, 16, 16, n, False, phase_train, scope='group_1')
        y = end_point['group_2'] = residual_group(y, 16, 32, n, True, phase_train, scope='group_2')
        y = end_point['group_3'] = residual_group(y, 32, 64, n, True, phase_train, scope='group_3')
        y = end_point['conv_last'] = conv2d(y, 64, n_classes, 1, 1, 'SAME', True, scope='conv_last')
        y = end_point['avg_pool'] = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
        y = end_point['squeeze'] = tf.squeeze(y, squeeze_dims=[1, 2])
    return y, end_point
