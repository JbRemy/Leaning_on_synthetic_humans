'''
Utilities for the networks
'''

import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from numpy.random import choice
from scipy.io import loadmat
from scipy.misc import imread
from scipy.ndimage.interpolation import rotate
import os



def make_batch(X_list, batch_size, input_H, input_W, output_dim, set='train'):
    '''
    Creats a batch to be fed to the training algorithm

    :param X_list: (.txt file) names of the training images
    :param batch_size: (int) batch size
    :param input_H: (int) Height of input images
    :param input_W: (int) Width of input images
    :param output_dim: (int) number of joints to compute
    :param set: (str) 'train', 'test' or 'val'

    :return: (tensor) (tensor)
    '''

    img_path = 'Data/{}/images'.format(set)
    mat_path = 'Data/{}/matrix'.format(set)
    X = np.zeros([batch_size, input_H, input_W, 3])
    y = np.zeros([batch_size, input_H, input_W, output_dim])
    with open(X_list, 'r') as file:
        lines = choice(file.readlines(), size=batch_size)

    for _ in lines:
        X[_, :, :, :], y[_, :, :, :] = preprocessing('{0}/{1}.mp4'.format(img_path, _), '{0}/{1}.npy'.format(mat_path, _),
                                                        input_H, input_W)

    return X, y


def compute_loss(input, y):
    '''
    computes the mean of cross entropy loss accross each heatmap output by the network

    :param input: (tensor)
    :param y: (tensor)

    :return: (float)
    '''

    loss = 0
    for stack in range(tf.shape(y)[0]):
        for n_heat_map in range(tf.shape(4)):
            input_heat_map = input[stack, :, :, :, n_heat_map]
            y_heat_map = y[:, :, :, n_heat_map]
            flat_input = flatten(input_heat_map)
            flat_y = flatten(y_heat_map)
            loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_input, labels=flat_y))

    loss = loss / (tf.shape(input)[0] + tf.shape(input)[1])

    return loss


rotate_scope = [-i for i in range(1, 10)] + [i for i in range(10)]
def preprocessing(img_path, mat_path, input_H, input_W):
    '''

    :param img_path:
    :param mat_path:
    :param input_H:
    :param input_W:
    :return:
    '''

    img = imread(img_path)
    joints = np.load(mat_path)
    angle = choice(rotate_scope)
    rotated_img, rotated_joints = rotate_image(img, joints, angle)


    return 0, 0


def rotate_image(img, joints, angle):
    '''
    rotates image and joints

    :param img: (np array) RGB chanels
    :param joints: (np array) matrix of joints
    :param angle: (int) angle of rotation in degrees

    :return: (np array) (np array)
    '''

    out_img = rotate(img, angle=angle, reshape=True)
    theta = 2 * np.pi * angle / 360
    out_joints = joints.copy()
    for _ in range(joints.shape[1]):
        x = joints[0, _] - 160
        y = joints[1, _] - 120
        if x > 0:
            theta_joint = np.arctan(y / x)
        elif x < 0:
            theta_joint = np.arctan(y / x) + np.pi
        elif x == 0:
            if y > 0:
                theta_joint = np.pi / 2
            else:
                theta_joint = -np.pi / 2

        r = np.sqrt(x ** 2 + y ** 2)
        out_joints[0, _] = r * np.cos(theta_joint - theta) + np.int(out_img.shape[1] / 2)
        out_joints[1, _] = r * np.sin(theta_joint - theta) + np.int(out_img.shape[0] / 2)

    return out_img, out_joints