import tensorflow as tf

label_ph = tf.placeholder(tf.float32, [None, 10])

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


# tensorboard
# 生成变量监控信息并定义生成监控信息日志的操作。其中var给出了需要记录的张量，name给出了
# 在可视化结果中显示的图标名称，这个名称一般与变量名一致。
def variable_summaries(var, name):
    # 将生成监控信息的操作放到同一个命名空间下。
    with tf.name_scope('summaries'):
        # 通过tf.summary.histogram函数记录张量中元素的取值分布。对于给出的图表
        # 名称和张量，tf.summary.histogram函数会生成一个Summary protocol buffer。
        # 将Summary写入TensorBoard日志文件后，在HISTOGRAMS栏和DISTRIBUTION栏
        # 下都会出现对应名称的图标。和TensorFlow中其他操作类似。tf.summary.histogram
        # 函数不会立刻被执行，只有当sess.run函数明确调用这个操作时，TensorFlow才会真正
        # 生成并输出Summary protocol buffer。下文将更加详细地介绍如何理解HISTOGRAMS栏
        # 和DISTRIBUTION栏下的信息。
        tf.summary.histogram(name, var)
        # 计算变量的平均值，并定义间生成平均值信息日志的操作。记录变量平均值信息的日志标签名
        # 为'mean/' + name，其中mean为命名空间，/是命名空间的分隔符。从图中可以看出，在相同
        # 命名空间中的监控指标会被整合到同一栏中。name则给出了当前监控指标属于哪一个变量。
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算变量的标准差，并定义生成其日志的操作。
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev/' + name, stddev)
