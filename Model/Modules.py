'''
Implementation of a stacked hourglass network as described in
"Stacked Hourglass Networks for Human Pose Estimation"  A. Newell, K. Yang, J. Deng
'''

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import conv2d

def make_Residual(input, n_feats_1=128, n_ouput_feats=256):
    '''
    Builds residual module of size :
    1*1*n_feats_1 -> 3*3*n_feats_1 ->  1*1*n_output_feats

    For computational purposes each layer is preceded by a batch_norm

    :param input: (tensor)
    :param n_feats_1: (int) the number of features to output from the first layer
    :param n_ouput_feats: (int) number of output features
    :return: list.txt, a list of all images in the created data set
    '''
    paddings = tf.constant([[1, 1, ], [2, 2]])

    norm_1 = batch_norm(input)
    conv_1 = conv2d(norm_1, n_feats_1, kernel_size=1)
    norm_2 = batch_norm(conv_1)
    conv_2 = conv2d(norm_2, n_feats_1, kernel_size=3)
    norm_3 = batch_norm(conv_2)
    conv_3 = conv2d(norm_3, int(numOut), kernel_size=1)

    return conv_3

def make_hourglass(input, =128, n_ouput_feats=256):

