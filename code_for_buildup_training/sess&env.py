import tensorflow as tf

# sess and global step
sess_config = tf.ConfigProto()
# 百分比
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
# 动态
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
global_steps = tf.Variable(0, trainable=False)

# env.
import os

# set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# about log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 默认值，打印所有信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 屏蔽INFO信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽INFO与WARNING信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 屏蔽INFO, WARING, ERROR信息

# pycharm2shell
import sys
sys.path.append('../')
