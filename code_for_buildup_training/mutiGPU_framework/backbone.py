# MNIST and Cifar10
import tensorflow as tf
import nets.lenet as lenet
import nets.cifarnet as cifarnet
import tensorflow.contrib.slim as slim

# tf.get_variable_scope()
def lenet_net(image, reuse=tf.AUTO_REUSE, keep_prop=0.5, is_train=True):
    image = tf.reshape(image, [-1, 28, 28, 1])
    with tf.variable_scope(name_or_scope='LeNet', reuse=reuse):
        arg_scope = lenet.lenet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = lenet.lenet(image, 10, is_training=is_train, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def cifar_net(image, reuse=tf.AUTO_REUSE, keep_prop=0.5, is_train=True):
    image = tf.reshape(image, [-1, 32, 32, 3])
    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=reuse):
        arg_scope = cifarnet.cifarnet_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_point = cifarnet.cifarnet(preprocessed, 10, is_training=is_train, dropout_keep_prob=keep_prop)
            probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


# ImageNet2012
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope
import nets.resnet as resnet


def inception(image, is_train=True, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    arg_scope = inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_point = inception_v3(preprocessed, 1001, is_training=is_train, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def resnet2_net(image, is_train=True, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    image = tf.reshape(preprocessed, [-1, 64, 64, 3])
    arg_scope = resnet.resnet_utils.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_point = resnet.resnet_v2_50(image, 1001, is_training=is_train, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point

def resnet101_net(image, is_train=True, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    image = tf.reshape(preprocessed, [-1, 64, 64, 3])
    arg_scope = resnet.resnet_utils.resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_point = resnet.resnet_v2_101(image, 1001, is_training=is_train, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point
