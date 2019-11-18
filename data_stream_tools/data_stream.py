import tensorflow.examples.tutorials.mnist.input_data as input_data
import pickle
import numpy as np
import tensorflow as tf
import os
import tensorflow.contrib.slim as slim


PATH_MNIST = "/home/kirin/Python_Code/#DATASET/MNIST"
PATH_CIFAR = "/home/kirin/Python_Code/#DATASET/Cifar/cifar-10-batches-py"
PATH_IMAGENET = "/mnt/exdata2/ImageNet_tfrecord/tfrecord"


class MNIST_datastream:
    def __init__(self, sess, batchsize=128, augmentation=False):
        self.mnist = input_data.read_data_sets(PATH_MNIST, one_hot=True)
        self.sess = sess

        features_placeholder = tf.placeholder(tf.float32, [None, 784])
        img_placeholder = tf.reshape(features_placeholder, [-1, 28, 28, 1])
        labels_placeholder = tf.placeholder(tf.int32, [None, 10])

        self.train_iterator = tf.data.Dataset.from_tensor_slices((img_placeholder, labels_placeholder)).shuffle(
            1).batch(batchsize).repeat().make_initializable_iterator()
        self.train_data = self.train_iterator.get_next()

        self.test_iterator = tf.data.Dataset.from_tensor_slices((img_placeholder, labels_placeholder)).shuffle(
            1).batch(batchsize).repeat().make_initializable_iterator()
        self.test_data = self.test_iterator.get_next()

        # init
        total_train_img = np.vstack((self.mnist.train.images, self.mnist.validation.images))
        total_train_label = np.vstack((self.mnist.train.labels, self.mnist.validation.labels))
        sess.run(self.train_iterator.initializer,
                 feed_dict={features_placeholder: total_train_img, labels_placeholder: total_train_label})

        sess.run(self.test_iterator.initializer,
                 feed_dict={features_placeholder: self.mnist.test.images, labels_placeholder: self.mnist.test.labels})


    def get_train_batch(self):
        image, label = self.sess.run([self.train_data[0], self.train_data[1]])
        return image, label

    def get_test_batch(self):
        image, label = self.sess.run([self.test_data[0], self.test_data[1]])
        return image, label


class Cifar_datastream:
    def __init__(self, sess, batchsize=128, augmentation=True):
        self.sess = sess

        self.IMAGE_SIZE = 32

        train_cifar_image = []
        train_cifar_label = []
        train_cifar_image_name = []

        for i in range(1, 6):
            cifar_file = PATH_CIFAR + "/data_batch_" + str(i)
            print("Reading...{}".format("data_batch_" + str(i)))
            cifar = self.unpickle(cifar_file)

            cifar_label = cifar[b'labels']
            cifar_image = cifar[b'data']
            cifar_image_name = cifar[b'filenames']
            cifar_image = cifar_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

            cifar_label = np.array(cifar_label)
            cifar_image_name = np.array(cifar_image_name)

            train_cifar_image.extend(cifar_image)
            train_cifar_label.extend(cifar_label)
            train_cifar_image_name.extend(cifar_image_name)

        test_cifar_image = []
        test_cifar_label = []
        test_cifar_image_name = []

        print("Reading...{}".format("test_set"))
        cifar_file = PATH_CIFAR + "/test_batch"
        cifar = self.unpickle(cifar_file)

        cifar_label = cifar[b'labels']
        cifar_image = cifar[b'data']
        cifar_image_name = cifar[b'filenames']
        cifar_image = cifar_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")

        test_cifar_image.extend(cifar_image)
        test_cifar_label.extend(cifar_label)
        test_cifar_image_name.extend(cifar_image_name)

        img_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels_placeholder = tf.placeholder(tf.int32, [None])

        if augmentation:
            # img = tf.image.resize_image_with_crop_or_pad(img_placeholder, self.IMAGE_SIZE + 4, self.IMAGE_SIZE + 4)
            # img = tf.map_fn(lambda x: tf.image.random_crop(x, [self.IMAGE_SIZE, self.IMAGE_SIZE, 3]), img, parallel_iterations=50000)
            img = tf.image.random_flip_left_right(img_placeholder)
            # img = tf.image.random_brightness(img, max_delta=0.1)
            # img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
            self.train_data_img = (img / 255 - 0.5) * 2
        else:
            self.train_data_img = (img_placeholder / 255 - 0.5) * 2


        self.test_data_img = (img_placeholder / 255 - 0.5) *2

        self.train_iterator = tf.data.Dataset.from_tensor_slices(
            (self.train_data_img, tf.one_hot(labels_placeholder, 10))).shuffle(
            20).batch(batchsize).repeat().make_initializable_iterator()
        self.train_data = self.train_iterator.get_next()

        self.test_iterator = tf.data.Dataset.from_tensor_slices(
            (self.test_data_img, tf.one_hot(labels_placeholder, 10))).shuffle(
            1).batch(batchsize).repeat().make_initializable_iterator()
        self.test_data = self.test_iterator.get_next()


        sess.run(self.train_iterator.initializer,
                 feed_dict={img_placeholder: train_cifar_image, labels_placeholder: train_cifar_label})

        sess.run(self.test_iterator.initializer,
                 feed_dict={img_placeholder: test_cifar_image, labels_placeholder: test_cifar_label})


    def unpickle(self, file):
        with open(file, 'rb') as f:
            cifar_dict = pickle.load(f, encoding='bytes')
        return cifar_dict

    def get_train_batch(self):
        image, label = self.sess.run([self.train_data[0], self.train_data[1]])
        return image, label

    def get_test_batch(self):
        image, label = self.sess.run([self.test_data[0], self.test_data[1]])
        return image, label


class ImageNet_datastream:
    def __init__(self, sess, batchsize=10, imgsize=224, just_eval=False):
        self.sess = sess
        self.val_img_batch, self.val_label_batch = self.read_and_decode(PATH_IMAGENET, "val", batchsize, imgsize)
        if not just_eval:
            self.train_img_batch, self.train_label_batch = self.read_and_decode(PATH_IMAGENET, "train", batchsize, imgsize)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)

        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94

    def read_and_decode(self, path, type="train", batchsize=10, imgsize=224):
        means = [123, 116, 103]
        if type == "train":
            file_path = os.path.join(path, "train-*")
            num_samples = 1281167

            dataset = self.get_record_dataset(file_path, num_samples=num_samples, num_classes=1000)
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=128)
            image, label = data_provider.get(['image', 'label'])

            image = self._fixed_sides_resize(image, output_height=imgsize, output_width=imgsize)

            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

            # method 1
            # channels = tf.split(axis=3, num_or_size_splits=3, value=image)
            # for i in range(3):
            #     channels[i] -= means[i]
            # image = tf.concat(axis=3, values=channels)/255

            # method 2
            image = 2 * (image / 255 - 0.5)

            img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batchsize, num_threads=128,
                                                            capacity=8192*3, min_after_dequeue=8192*2)
            label_batch = tf.one_hot(label_batch-1, 1000)
            return img_batch, label_batch
        else:
            file_path = os.path.join(path, "validation-*")
            num_samples = 50000

            dataset = self.get_record_dataset(file_path, num_samples=num_samples, num_classes=1000)
            data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
            image, label = data_provider.get(['image', 'label'])

            image = self._fixed_sides_resize(image, output_height=imgsize, output_width=imgsize)

            # method 1
            # channels = tf.split(axis=2, num_or_size_splits=3, value=image)
            # for i in range(3):
            #     channels[i] -= means[i]
            # image = tf.concat(axis=2, values=channels)/255

            # method 2
            image = 2 * (image / 255 - 0.5)

            img_batch, label_batch = tf.train.batch([image, label], batch_size=batchsize, allow_smaller_final_batch=True)
            # img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batchsize, num_threads=16,
            #                                                 capacity=4096, min_after_dequeue=512)
            label_batch = tf.one_hot(label_batch-1, 1000)
            return img_batch, label_batch


    def _fixed_sides_resize(self, image, output_height, output_width):
        """Resize images by fixed sides.

        Args:
            image: A 3-D image `Tensor`.
            output_height: The height of the image after preprocessing.
            output_width: The width of the image after preprocessing.
        Returns:
            resized_image: A 3-D tensor containing the resized image.
        """
        output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
        output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_nearest_neighbor(
            image, [output_height, output_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    def get_record_dataset(self, record_path, reader=None, num_samples=1281167, num_classes=1000):
        """Get a tensorflow record file.

        Args:

        """
        if not reader:
            reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/class/label':
                tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1],
                                                                         dtype=tf.int64))}

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                                  format_key='image/format'),
            'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        items_to_descriptions = {
            'image': 'An image with shape image_shape.',
            'label': 'A single integer.'}
        return slim.dataset.Dataset(
            data_sources=record_path,
            reader=reader,
            decoder=decoder,
            num_samples=num_samples,
            num_classes=num_classes,
            items_to_descriptions=items_to_descriptions,
            labels_to_names=labels_to_names)

    def get_train_batch(self):
        image, label = self.sess.run([self.train_img_batch, self.train_label_batch])
        return image, label

    def get_test_batch(self):
        image, label = self.sess.run([self.val_img_batch, self.val_label_batch])
        return image, label

