import tensorflow as tf
import numpy as np

sess = tf.Session()
global_steps = tf.Variable(0, trainable=False)


# fast feed
cleanimg_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
label_ph = tf.placeholder(tf.float32, [None, 10])
keepprob_ph = tf.placeholder(tf.float32)

def get_feed(batch_image, batch_label, type="train"):
    if type == "train":
        return {cleanimg_ph: batch_image, label_ph: batch_label, keepprob_ph: 0.5}
    else:
        return {cleanimg_ph: batch_image, label_ph: batch_label, keepprob_ph: 1.0}



# load target var
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/') and "RMSProp" not in var.name
]
pre_train_saver = tf.train.Saver(restore_vars)


# train op setting
total_loss = 0
lr = 1
subdivisions = 10

# common train
train_op = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1.0).minimize(total_loss)


# 手动赋值梯度
# 创建一个optimizer.
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 计算<list of variables>相关的梯度
var = tf.trainable_variables()
grads_and_vars = opt.compute_gradients(total_loss, var)
# grads_and_vars为tuples (gradient, variable)组成的列表。
# 令optimizer运用capped的梯度(gradients)
opt.apply_gradients(grads_and_vars)


# sym. traning on single GPU
# 梯度保存实现单机超大batch训练
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


# 多GPU常用的代码片段
num_gpus = 8
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
 
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device 
    return _assign

with tf.device("/cpu:0"):
    tower_grads = []
    X = tf.placeholder(tf.float32, [None, 10])
    Y = tf.placeholder(tf.float32, [None, 10])
    opt = tf.train.AdamOptimizer(lr)
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(num_gpus):
            with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                # _x = X[i * batch_size:(i + 1) * batch_size]
                # _y = Y[i * batch_size:(i + 1) * batch_size]
                # logits = conv_net(_x, True)
                tf.get_variable_scope().reuse_variables()
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=logits))
                grads = opt.compute_gradients(loss)
                tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    train_op = opt.apply_gradients(grads)


# 使用batchnorm可能会用到的代码段
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = opt.minimize(loss)

var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=5)