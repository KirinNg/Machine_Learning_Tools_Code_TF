import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

def apply_with_random_selector(x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image):
    image = image/255.
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)

    # Randomly distort the colors. There are 1 or 4 ways to do it.
    num_distort_cases = 1
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering),
        num_cases=num_distort_cases)

    distorted_image = tf.multiply(distorted_image, 255.0)
    return distorted_image


def Get_ALL_Loss(model_fn, input_image, input_label):
    Chirality_img = preprocess_for_train(input_image)

    logits, probs, end_point = model_fn(input_image)

    C_logits, C_probs, C_end_point = model_fn(Chirality_img)

    # closs
    # closs = slim.losses.softmax_cross_entropy(logits, input_label)
    closs = (tf.losses.softmax_cross_entropy(input_label, logits) + tf.losses.softmax_cross_entropy(input_label, C_logits))/2

    # l2 loss
    # try:
    l2_loss = tf.add_n(slim.losses.get_regularization_losses())
    # except:
        # l2_loss = tf.add_n([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
        # l2_loss = tf.zeros()

    # Chirality loss
    chirality_loss = tf.losses.mean_squared_error(logits, C_logits)
    # chirality_loss = slim.losses.mean_squared_error(C_logits, logits)

    # eval
    correct_p = tf.equal(tf.argmax(logits, 1), (tf.argmax(input_label, 1)))
    tmp_accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))

    LOSS_DIR = {'closs': closs,
                'l2_loss': l2_loss,
                'chirality_loss': chirality_loss,
                'ori_img': input_image,
                'clean_acc': tmp_accuracy,}

    return LOSS_DIR

