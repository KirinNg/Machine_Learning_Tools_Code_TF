import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import PIL
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2

sess = tf.InteractiveSession()

image = tf.placeholder(tf.float32, [1, 299, 299, 3])


def inception(image, reuse=tf.AUTO_REUSE):
    preprocessed = tf.multiply(tf.subtract(image / 255, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.inception.inception_v3(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def step_target_class_adversarial_images(x, eps, one_hot_target_class):
    logits, _, end_points = inception(x, reuse=True)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    cross_entropy += tf.losses.softmax_cross_entropy(one_hot_target_class,
                                                     end_points['AuxLogits'][:, 1:],
                                                     label_smoothing=0.1,
                                                     weights=0.4)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, 0, 1.0)
    return tf.stop_gradient(x_adv)


def stepll_adversarial_images(x, eps):
    logits, _, _ = inception(x, reuse=True)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    return step_target_class_adversarial_images(x, eps, one_hot_ll_class)


def stepllnoise_adversarial_images(x, eps):
    logits, _, _ = inception(x, reuse=True)
    least_likely_class = tf.argmin(logits, 1)
    one_hot_ll_class = tf.one_hot(least_likely_class, 1000)
    x_noise = x + eps / 2 * tf.sign(tf.random_normal(x.shape))
    return step_target_class_adversarial_images(x_noise, eps / 2,
                                                one_hot_ll_class)


logits, probs, end_point = inception(image)
origin_label = tf.argmax(logits, axis=1)
adv_image = stepll_adversarial_images(image, 0.01)

num_class = 1000
layer_name = 'Mixed_7c'
conv_layer = end_point[layer_name]
pre_calss = tf.placeholder(tf.int32)
one_hot = tf.sparse_to_dense(pre_calss, [num_class], 1.0)
signal = tf.multiply(end_point['Logits'][:, 1:], one_hot)
loss = tf.reduce_mean(signal)
grads = tf.gradients(loss, conv_layer)[0]
norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))


def grad_cam(x, class_num):
    output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={image: [x], pre_calss: class_num})
    output = output[0]
    grads_val = grads_val[0]
    weights = np.mean(grads_val, axis=(0, 1))  # [512]
    cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]
    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    # Passing through ReLU
    cam = cam - 10
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam3 = np.expand_dims(cam, axis=2)
    cam = np.tile(cam3, [1, 1, 3])
    cam = resize(cam, (299, 299, 3))
    # plt.imshow(cam)
    # plt.show()
    return cam


def show_img(img, heatmap):
    img = cv2.resize(img, (299, 299))
    img = img.astype(float)
    img /= img.max()
    rar = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    rar = cv2.cvtColor(rar, cv2.COLOR_BGR2RGB)
    alpha = 0.0072
    rar = img + alpha * rar
    rar /= rar.max()
    plt.imshow(rar)
    plt.show()

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]

saver = tf.train.Saver(restore_vars)
saver.restore(sess, "/home/kirin/Python_Code/ResNet100/inception_v3.ckpt")


def load_img(path):
    I = PIL.Image.open(path)
    I = I.resize((299, 299)).crop((0, 0, 299, 299))
    I = (np.asarray(I) / 255.0).astype(np.float32)
    return I


if __name__ == '__main__':
    I_input = load_img("/home/kirin/Python_Code/ResNet100/tank.jpg")
    label_of_img = sess.run(origin_label, feed_dict={image: [I_input]})
    G_cam = grad_cam(I_input, label_of_img)
    show_img(I_input, G_cam)
    adv = I_input
    for i in range(30):
        adv = sess.run(adv_image, feed_dict={image: [adv]})
        adv = adv[0]
    plt.imshow(adv)
    plt.show()
    Adv_G_cam = grad_cam(adv, label_of_img)
    show_img(I_input, Adv_G_cam)
