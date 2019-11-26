import tensorflow as tf
import nets.lenet as lenet
import nets.cifarnet as cifarnet
import nets.vgg as vgg
import nets.resnet as resnet
import nets.alexnet as alexnet
import nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim as slim
from useful_networks.preprocess.vgg import *
from useful_networks.preprocess.inception import *
from useful_networks.preprocess.cifar import *


def lenet_net(image, reuse=tf.AUTO_REUSE, keep_prop=0.5, is_train=True):
    image = tf.reshape(image, [-1, 28, 28, 1])
    with tf.variable_scope(name_or_scope='', reuse=reuse):
        arg_scope = lenet.lenet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = lenet.lenet(image, 10, is_training=is_train, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def cifar_net(image, reuse=tf.AUTO_REUSE, keep_prop=0.5, is_train=True):
    image = tf.reshape(image, [-1, 32, 32, 3])
    with tf.variable_scope(name_or_scope='', reuse=reuse):
        arg_scope = cifarnet.cifarnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = cifarnet.cifarnet(image, 10, is_training=is_train, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def cifair_rsenet_20(image, reuse=tf.AUTO_REUSE, if_train=True):
    # 32 layer
    with tf.variable_scope(name_or_scope='ResNet', reuse=reuse):
        image = tf.reshape(image, [-1, 32, 32, 3])
        logits, end_point = resnet_v1.residual_net(image, 3, 10, if_train, reuse=reuse)
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def cifair_rsenet_32(image, reuse=tf.AUTO_REUSE, if_train=True):
    # 32 layer
    with tf.variable_scope(name_or_scope='ResNet', reuse=reuse):
        image = tf.reshape(image, [-1, 32, 32, 3])
        logits, end_point = resnet_v1.residual_net(image, 5, 10, if_train)
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def cifair_rsenet_56(image, reuse=tf.AUTO_REUSE, if_train=True):
    # 56 layer
    with tf.variable_scope(name_or_scope='ResNet', reuse=reuse):
        image = tf.reshape(image, [-1, 32, 32, 3])
        logits, end_point = resnet_v1.residual_net(image, 9, 10, if_train)
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def Alex_net(image, reuse=tf.AUTO_REUSE, keep_prop=0.5):
    image = tf.reshape(image, [-1, 224, 224, 3])
    with tf.variable_scope(name_or_scope='', reuse=reuse):
        arg_scope = alexnet.alexnet_v2_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = alexnet.alexnet_v2(image, 1000, is_training=True, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def VGG16(image, reuse=tf.AUTO_REUSE):
    preprocess = lambda x: preprocess_image(x, 224, 224, is_training=False)
    preprocessed = tf.map_fn(preprocess, elems=image)
    # preprocessed = preprocess_for_eval(image, 224, 224, 256)
    arg_scope = vgg.vgg_arg_scope(weight_decay=0.0)
    with tf.variable_scope(name_or_scope='', reuse=reuse):
        with slim.arg_scope(arg_scope):
            logits, end_point = vgg.vgg_16(preprocessed, 1000, is_training=False, dropout_keep_prob=1.0)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def resnet2_net(image, reuse=tf.AUTO_REUSE):
    image = tf.reshape(image, [-1, 224, 224, 3])
    with tf.variable_scope(name_or_scope='', reuse=reuse):
        arg_scope = resnet.resnet_utils.resnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = resnet.resnet_v2_50(image, 1000, is_training=True)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point
