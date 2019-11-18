import tensorflow as tf
import numpy as np


# load target var
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/') and "RMSProp" not in var.name
]
pre_train_saver = tf.train.Saver()


# sess and global step
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
global_steps = tf.Variable(0, trainable=False)


# Advanced Learning rate
epoch_per_step = 1
lr_decay = tf.train.exponential_decay(0.045, global_steps, 2*epoch_per_step, 0.94)

boundaries = [20000*3, 50000*3]
learing_rates = [0.01, 0.001, 0.0001]
lr_split = tf.train.piecewise_constant(global_steps, boundaries=boundaries, values=learing_rates)


# sym. traning on single GPU
total_loss = 0
lr = 1
subdivisions = 10

train_op = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1.0)
grads_vars = train_op.compute_gradients(total_loss, tf.trainable_variables())
for i in range(len(grads_vars))[::-1]:
    if grads_vars[i][0] is None:
        del grads_vars[i]
grads_cache = [tf.Variable(np.zeros(t[0].shape.as_list(), np.float32), trainable=False) for t in grads_vars]
clear_grads_cache_op = tf.group([gc.assign(tf.zeros_like(gc)) for gc in grads_cache])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    accumulate_grad_op = tf.group([gc.assign_add(gv[0]) for gc, gv in zip(grads_cache, grads_vars)])
mean_grad = [tf.clip_by_value(gc/tf.to_float(subdivisions), -2.0, 2.0) for gc in grads_cache]
new_grads_vars = [(g, gv[1]) for g, gv in zip(mean_grad, grads_vars)]

apply_grad_op = train_op.apply_gradients(new_grads_vars, global_step=global_steps)

for i in range(1):
    sess.run(clear_grads_cache_op)
    for s in range(subdivisions):
        sess.run([accumulate_grad_op])
    sess.run(apply_grad_op)


# env.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# fast feed
cleanimg_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
label_ph = tf.placeholder(tf.float32, [None, 10])
keepprob_ph = tf.placeholder(tf.float32)

def get_feed(batch_image, batch_label, type="train"):
    if type == "train":
        return {cleanimg_ph: batch_image, label_ph: batch_label, keepprob_ph: 0.5}
    else:
        return {cleanimg_ph: batch_image, label_ph: batch_label, keepprob_ph: 1.0}


# eval in traning
# top N
logits = None
batch_size = 32
def caculate_topK(indices, k, batch_size=128):
    label = tf.argmax(label_ph, axis=1, output_type=tf.int32)
    a = indices - tf.reshape(label, (batch_size, 1))
    b = tf.equal(a, tf.zeros(shape=(batch_size, k), dtype=tf.int32))
    return tf.reduce_mean(tf.reduce_sum(tf.cast(b, tf.float32), axis=1), name='top_{}'.format(k))

_, top_5_indices = tf.nn.top_k(logits, k=5, name='top_5_indices')
acc_top_5 = caculate_topK(top_5_indices, 5, batch_size)

# acc
correct_p = tf.equal(tf.argmax(logits, 1), (tf.argmax(label_ph, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))


# cal time
import datetime

for e in range(5):
    begin_time = datetime.datetime.now()
    end_time = datetime.datetime.now()
    during_time = end_time - begin_time
    print(during_time.microseconds)


# trick for tqgm
import tqdm
pbar = tqdm.trange(iter)
for k in pbar:
    pbar.set_description("hello!")
