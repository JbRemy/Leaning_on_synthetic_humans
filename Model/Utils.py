'''
Utilities for the networks
'''

import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from numpy.random import choice
from scipy.misc import imread
from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize



def make_batch(X_list, batch_size, input_H=256, input_W=256, output_dim=13, set='train',
               adversarial=False, Source=True):
    '''
    Creats a batch to be fed to the training algorithm

    :param X_list: (.txt file) names of the training images
    :param batch_size: (int) batch size
    :param input_H: (int) Height of input images
    :param input_W: (int) Width of input images
    :param output_dim: (int) number of joints to compute
    :param set: (str) 'train', 'test' or 'val' 'real'
    :param adversarial: (Boolean)
    :param Source: (Boolean)

    :return: (tensor) (tensor)
    '''

    img_path = 'Data/{}/images'.format(set)
    mat_path = 'Data/{}/matrix'.format(set)
    X = np.zeros([batch_size, input_H, input_W, 3])
    y = np.zeros([batch_size, int(input_H/4), int(input_W/4), output_dim])

    with open(X_list, 'r') as file:
        lines = choice(file.readlines(), size=batch_size)

    for _ in range(len(lines)):
        X[_, :, :, :], y[_, :, :, :] = preprocessing('{0}/{1}.jpg'.format(img_path, lines[_].strip()),
                                                     '{0}/{1}.npy'.format(mat_path, lines[_].strip()))

    if adversarial:
        if Source:
            y_domain = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))], axis=1)

        else:
            y_domain = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))], axis=1)

        return X, y, y_domain

    else:

        return X, y


rotate_scope = [-i for i in range(1, 10)] + [i for i in range(10)]
def preprocessing(img_path, mat_path):
    '''
    loads images, randomly rotates it, crops it so that the image is squared and focused on the body then resize it.
    Applies same transformaiton to the corresponding joints

    :param img_path: (str) path to the image
    :param mat_path: (str) path to the corresponding joint

    :return: RGB, np array with one channel by joint (one hot)
    '''

    img = imread(img_path)
    joints = np.load(mat_path)
    angle = choice(rotate_scope)
    rotated_img, rotated_joints = rotate_image(img, joints, angle)
    croped_img, croped_joints = crop_image(rotated_img, rotated_joints)
    resized_img, resized_joints = resize_image_joints(croped_img, croped_joints)
    norm_img = resized_img / 255
    heat_maps = make_heat_maps(resized_joints)

    return norm_img, heat_maps


def rotate_image(img, joints, angle):
    '''
    rotates image and joints

    :param img: (np array) RGB chanels
    :param joints: (np array) position of the joints [2, N_joints]
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


def crop_image(img, joints):
    '''
    Crop the image and adjust joinnts position, so that the image becomes square and focused on the body.

    :param img: (np darray) RGB
    :param joints: (np array) position of the joints [2, N_joints]

    :return: (np array) (np array)
    '''

    x_dist = joints[1, :].max() - joints[1, :].min()
    y_dist = joints[0, :].max() - joints[0, :].min()
    dist = max(x_dist, y_dist) + 30
    mid_x = (joints[1, :].max() + joints[1, :].min()) / 2
    mid_y = (joints[0, :].max() + joints[0, :].min()) / 2
    x_min = np.int(mid_x - dist / 2)
    x_max = np.int(mid_x + dist / 2)
    y_min = np.int(mid_y - dist / 2)
    y_max = np.int(mid_y + dist / 2)
    if x_min < 0:
        x_max -= x_min
        x_min = 0

    if y_min < 0:
        y_max -= y_min
        y_min = 0

    if x_max > img.shape[0]-1:
        x_min -= x_max-img.shape[0]+1
        x_max = img.shape[0]-1

    if y_max > img.shape[1]-1:
        y_min -= y_max-img.shape[1]+1
        y_max = img.shape[1]-1

    img = img[x_min:x_max, y_min:y_max, :]
    joints[0, :] = joints[0, :] - y_min
    joints[1, :] = joints[1, :] - x_min

    return img, joints


def resize_image_joints(img, joints, img_size=256, joints_size=64):
    '''
    Crop the image and adjust joinnts position, so that the image becomes square and focused on the body.

    :param img: (np darray) RGB, needs to be square
    :param joints: (np array) position of the joints [2, N_joints]
    :param img_size: (int) the new size of the image
    :param joints_size: (int) output size for joints matrix

    :return: (np array) (np array)
    '''

    rate = (joints_size-1) / max(img.shape[0], 1)
    x_mid = img.shape[1] / 2
    y_mid = img.shape[0] / 2
    img = imresize(img, [img_size, img_size], interp='bilinear')
    for _ in range(joints.shape[1]):
        joints[0, _] = max(0, min(np.int((joints[0, _] - y_mid) * rate + joints_size / 2), joints_size-1))
        joints[1, _] = max(0, min(np.int((joints[1, _] - x_mid) * rate + joints_size / 2), joints_size-1))

    return img, joints


def make_heat_maps(joints, size=64):
    '''
    Computes heat maps for all joints

    :param joints: (np array) position of the joints [2, N_joints]
    :param size: (int) size of heatmap

    :return: (np array) a chanel by joint
    '''

    joints_mats = np.zeros([size, size, joints.shape[1]])
    for _ in range(joints.shape[1]):
        joints_mats[joints[0,_], joints[1, _], _] = 1

    return joints_mats


def compute_loss(input, y):
    '''
    computes the mean of cross entropy loss accross each heatmap output by the network

    :param input: (tensor)
    :param y: (tensor)

    :return: (float)
    '''

    loss_list = []
    for stack in range(input.get_shape().as_list()[0]):
        for n_heat_map in range(input.get_shape().as_list()[4]):
            input_heat_map = input[stack, :, :, :, n_heat_map]
            y_heat_map = y[:, :, :, n_heat_map]
            flat_input = flatten(input_heat_map)
            flat_y = flatten(y_heat_map)
            loss_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_input, labels=flat_y), name='loss'))

    loss = tf.add_n(loss_list) / (input.get_shape().as_list()[0] * input.get_shape().as_list()[4])

    return loss


def mat_to_joints(heat_maps):
    '''
    Computes joints from a set of heat maps

    :param heat_maps: (np array) [64, 64, out_dim]

    :return: (np array) [2, out_dim]
    '''

    joints = np.zeros([2, heat_maps.shape[2]])
    for _ in range(heat_maps.shape[2]):
        joints[:,_] = np.unravel_index(heat_maps[:, :, _].argmax(), heat_maps[:, :, _].shape)

    return joints
