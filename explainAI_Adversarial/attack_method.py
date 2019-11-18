import tensorflow as tf
import numpy as np
import cleverhans.utils_tf as utils_tf


def FGSM(x, logits, eps=0.15):
    total_class_num = tf.shape(logits)[1]
    ori_class = tf.argmax(logits, 1)
    one_hot_class = tf.one_hot(ori_class, total_class_num)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x + eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    # x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def target_FGSM(x, logits, target, eps=0.15):
    cross_entropy = tf.losses.softmax_cross_entropy(target,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x + eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    # x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def StepLL(x, logits, eps=0.15):
    total_class_num = tf.shape(logits)[1]
    ori_class = tf.argmin(logits, 1)
    one_hot_class = tf.one_hot(ori_class, total_class_num)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x - eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    # x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
    return tf.stop_gradient(x_adv)


def CW():
    pass


def Jsma():
    pass


def jacobian_graph(predictions, x, nb_classes):
  """
  Create the Jacobian graph to be ran later in a TF session
  :param predictions: the model's symbolic output (linear output,
      pre-softmax)
  :param x: the input placeholder
  :param nb_classes: the number of classes the model has
  :return:
  """
  # This function will return a list of TF gradients
  list_derivatives = []

  # Define the TF graph elements to compute our derivatives for each class
  for class_ind in range(nb_classes):
    derivatives, = tf.gradients(predictions[:, class_ind], x)
    list_derivatives.append(derivatives)
  return list_derivatives


def deepfool_attack(sess, x, predictions, logits, sample, nb_candidate=10, overshoot=0.03, max_iter=30, clip_min=0.0, clip_max=1.0, feed=None):
    """
    TensorFlow implementation of DeepFool.
    Paper link: see https://arxiv.org/pdf/1511.04599.pdf
    :param sess: TF session
    :param x: The input placeholder
    :param predictions: The model's sorted symbolic output of logits, only the
                       top nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                 from gradient_graph
    :param sample: Numpy array with sample input
    :param nb_candidate: The number of classes to test against, i.e.,
                        deepfool only consider nb_candidate classes when
                        attacking (thus accelerate speed)
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :return: an adversarial sample
    """
    import copy

    adv_x = copy.copy(sample)
    # Initialize the loop variables
    iteration = 0
    current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
    if current.shape == ():
        current = np.array([current])
    w = np.squeeze(np.zeros(sample.shape[1:4]))  # same shape as original image
    r_tot = np.zeros(sample.shape)
    original = current  # use original label as the reference

    grads = jacobian_graph(predictions, x, nb_candidate)

    # Repeat this main loop until we have achieved misclassification
    while (np.any(current == original) and iteration < max_iter):
        feed.update({x: adv_x})
        gradients = sess.run(grads, feed_dict=feed)
        predictions_val = sess.run(predictions, feed_dict=feed)
        for idx in range(sample.shape[0]):
            pert = np.inf
            if current[idx] != original[idx]:
                continue
            for k in range(1, nb_candidate):
                w_k = gradients[k][idx, ...] - gradients[0][idx, ...]
                f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                # adding value 0.00001 to prevent f_k = 0
                pert_k = (abs(f_k) + 1e-30) / np.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = pert*w/np.linalg.norm(w)
            r_tot[idx, ...] = r_tot[idx, ...] + r_i

        # adv_x = np.clip(r_tot + sample, clip_min, clip_max)
        adv_x = r_tot + sample
        feed.update({x: adv_x})

        current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
        if current.shape == ():
            current = np.array([current])
        # Update loop variables
        iteration = iteration + 1

    # need to clip this image into the given range
    # adv_x = np.clip((1+overshoot)*r_tot + sample, clip_min, clip_max)
    adv_x = (1 + overshoot) * r_tot + sample
    return adv_x

import copy

class deepfool_attack_batch:
    def __init__(self, sess, x, predictions, logits, sample, nb_candidate=10, overshoot=0.05, max_iter=10, feed=None):
        self.grads = jacobian_graph(predictions, x, nb_candidate)

    def get_deepfool(self, sess, x, predictions, logits, sample, nb_candidate=10, overshoot=0.03, max_iter=30, feed=None):
        adv_x = copy.copy(sample)
        # Initialize the loop variables
        current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
        if current.shape == ():
            current = np.array([current])
        w = np.squeeze(np.zeros(sample.shape[1:4]))  # same shape as original image
        r_tot = np.zeros(sample.shape)
        original = current  # use original label as the reference

        iteration = 0
        # Repeat this main loop until we have achieved misclassification
        while (np.any(current == original) and iteration < max_iter):
            feed.update({x: adv_x})
            gradients, predictions_val = sess.run([self.grads, predictions], feed_dict=feed)
            for idx in range(sample.shape[0]):
                pert = np.inf
                if current[idx] != original[idx]:
                    continue
                for k in range(1, nb_candidate):
                    w_k = gradients[k][idx, ...] - gradients[0][idx, ...]
                    f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                    # adding value 0.00001 to prevent f_k = 0
                    pert_k = (abs(f_k) + 1e-30) / np.linalg.norm(w_k.flatten())
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k
                r_i = pert*w/np.linalg.norm(w)
                r_tot[idx, ...] = r_tot[idx, ...] + r_i

            # adv_x = np.clip(r_tot + sample, clip_min, clip_max)
            adv_x = r_tot + sample
            feed.update({x: adv_x})

            current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
            if current.shape == ():
                current = np.array([current])
            # Update loop variables
            iteration = iteration + 1

        # need to clip this image into the given range
        # adv_x = np.clip((1+overshoot)*r_tot + sample, clip_min, clip_max)
        adv_x = (1 + overshoot) * r_tot + sample
        return adv_x

