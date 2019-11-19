import tensorflow as tf

global_steps = tf.Variable(0, trainable=False)

# Advanced Learning rate
epoch_per_step = 1
lr_decay = tf.train.exponential_decay(0.045, global_steps, 2*epoch_per_step, 0.94)

boundaries = [20000*3, 50000*3]
learing_rates = [0.01, 0.001, 0.0001]
lr_split = tf.train.piecewise_constant(global_steps, boundaries=boundaries, values=learing_rates)
