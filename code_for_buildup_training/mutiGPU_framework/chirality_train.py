import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import tqdm
import matplotlib.pyplot as plt
import numpy as np
import collections
import six
import itertools
from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

from chirality_train.Loss_Block import *
from chirality_train.data_stream import *
from chirality_train.backbone import *


sess_config = tf.ConfigProto(allow_soft_placement=True)

sess_config.gpu_options.allow_growth = True
sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=sess_config)

model_fn = resnet2_net
num_gpus = 4

# data
DATA_STREAM = ImageNet_datastream(sess, 128 * num_gpus, imgsize=224)
train_img, train_label = DATA_STREAM.get_train_batch()

lr_rate = 1e-6

tower_losses = []
tower_gradvars = []


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']
        # ps_ops = ['Variable', 'VariableV2', 'VarHandleOp', 'Const', 'Fill', 'Assign', 'Identity', 'ApplyAdam']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string('/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


with tf.device("/cpu:0"):
    split_train_img = tf.split(train_img, num_gpus)
    split_train_label = tf.split(train_label, num_gpus)
    update_ops = 0
    for i in range(num_gpus):
        worker_device = '/{}:{}'.format('gpu', i)
        device_setter = local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
        with tf.device(device_setter):
            Loss_DIC = Get_ALL_Loss(model_fn, split_train_img[i], split_train_label[i])

            Loss = Loss_DIC['closs']
            Loss += Loss_DIC['l2_loss']

            model_params = tf.trainable_variables()
            grads = tf.gradients(Loss, model_params)
            tower_gradvars.append(zip(grads, model_params))
            tower_losses.append(Loss)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in six.iteritems(all_grads):
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))

# Device that runs the ops to apply global gradient updates.
consolidation_device = '/cpu:0'
with tf.device(consolidation_device):
    optimizer = slim.train.AdamOptimizer(learning_rate=lr_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(gradvars, global_step=tf.train.get_global_step())
    Loss = tf.reduce_mean(tower_losses)

# eval
val_img, val_label = DATA_STREAM.get_test_batch()

with tf.device("/cpu:0"):
    tower_acc = []
    tower_val_acc = []

    val_split_img = tf.split(val_img, num_gpus)
    val_split_label = tf.split(val_label, num_gpus)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            worker_device = '/{}:{}'.format('gpu', i)
            device_setter = local_device_setter(
                ps_device_type='gpu',
                worker_device=worker_device,
                ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                    num_gpus, tf.contrib.training.byte_size_load_fn))

            with tf.device(device_setter):
                # eval
                logits, probs, end_point = model_fn(val_split_img[i], is_train=False)
                correct_p = tf.equal(tf.argmax(logits, 1), (tf.argmax(val_split_label[i], 1)))
                tmp_accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
                tower_acc.append(tmp_accuracy)

    val_clean_accuracy = tf.reduce_mean(tower_acc)

sess.run(tf.global_variables_initializer())

variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['.*/Adam', 'beta1_power', 'beta2_power'])
# variables_to_restore = tf.contrib.framework.get_variables_to_restore()
saver_load = tf.train.Saver(variables_to_restore)
saver_save = tf.train.Saver(tf.global_variables())

sess.graph.finalize()

MAIN_ITER = 1000

saver_save.restore(sess, "models/resnet50_imagenet.ckpt")

train_bar = tqdm.trange(MAIN_ITER)
for i in train_bar:
    sess.run([train_op, Loss_DIC['closs'], Loss_DIC['chirality_loss'], Loss_DIC['l2_loss']])
    train_bar.set_description(str([]))
    if i % 1000 == 999:
        saver_save.save(sess, "models/resnet50_imagenet.ckpt")

saver_save.save(sess, "models/resnet50_imagenet.ckpt")

Iter_NUM = 100

Final_acc = 0
pbar = tqdm.trange(Iter_NUM)
for i in pbar:
    tmp_acc = sess.run(val_clean_accuracy)
    Final_acc += tmp_acc
    pbar.set_description("clean_Final_acc:{:.2f}".format(Final_acc / (i + 1) * 100))
print("clean_Final_acc:{:.2f}".format(Final_acc / Iter_NUM * 100))

