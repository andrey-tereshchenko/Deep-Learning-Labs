import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


def linear_correction(img):
    image = tf.image.rgb_to_hsv(img)
    img_min = tf.reduce_min(tf.reduce_min(image, axis=0), axis=0)
    img_max = tf.reduce_max(tf.reduce_max(image, axis=0), axis=0)
    img_max = tf.where(img_max == 0, 0.1 * tf.ones_like(img_max), img_max)
    image = tf.math.divide(tf.math.subtract(image, img_min),
                           tf.math.subtract(img_max, img_min)) * tf.constant([1., 1., 255.])
    image = tf.image.hsv_to_rgb(image)
    return image


def exponential_correction(c, p, img):
    c_tensor = tf.constant([1.0, 1.0, c])
    p_tensor = tf.constant([1.0, 1.0, p])
    image = tf.image.rgb_to_hsv(img)
    image = tf.math.multiply(image, c_tensor)
    image = tf.math.pow(image, p_tensor)
    image = tf.image.hsv_to_rgb(image)
    return image


def get_gaussian_kernel(size, mean, std):
    d = tf.compat.v1.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def get_box_filter_kernel(size):
    box_filter = tf.constant([[1 / (size ** 2) for i in range(size)] for j in range(size)])
    return box_filter


def get_unsharp_masking_kernel(l):
    first_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    second_matrix = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    unsharp_masking_kernel = first_matrix + l * second_matrix
    return tf.convert_to_tensor(unsharp_masking_kernel)


def get_sobel_kernel_x():
    sobel_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return tf.convert_to_tensor(sobel_kernel)


def get_sobel_kernel_y():
    sobel_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return tf.convert_to_tensor(sobel_kernel)


select = int(input('Input - 1 for image correction, Input - 2 for image convolution: '))
img = cv2.imread('images/3.jpg')
img = tf.convert_to_tensor(img)[..., ::-1]
img = tf.dtypes.cast(img, tf.float32)
if select == 1:
    type_correction = int(input('Input 1 - linear, 2 - exponential: '))
    if type_correction == 1:
        img = linear_correction(img)
    else:
        img = exponential_correction(0.9, 1, img)
else:
    kernel_type = int(input('Input 1 - gaussian, 2 - box, 3 - unsharp, 4 - sobel: '))
    if kernel_type == 1:
        kernel_2d = get_gaussian_kernel(8, 0.0, 4.0)
    elif kernel_type == 2:
        kernel_2d = get_box_filter_kernel(10)
    elif kernel_type == 3:
        kernel_2d = get_unsharp_masking_kernel(2)
    else:
        kernel_2d = get_sobel_kernel_x()
    kernel_2d = tf.dtypes.cast(kernel_2d, tf.float32)
    kernel = tf.tile(kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 1])
    point_wise_filter = tf.eye(3, batch_shape=[1, 1])
    image = tf.nn.separable_conv2d(tf.expand_dims(img, 0), kernel, point_wise_filter,
                                   strides=[1, 1, 1, 1], padding='SAME')
    img = tf.squeeze(image)
img = tf.where(img < 0, 0, img)
img = tf.dtypes.cast(img, tf.uint32)

plt.figure(figsize=(12, 9))
plt.imshow(img.numpy())
plt.show()
