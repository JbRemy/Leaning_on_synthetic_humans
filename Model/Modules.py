'''
Modules to implement an domain adversarial version of the stacked hourglass network described in
"Stacked Hourglass Networks for Human Pose Estimation"  A. Newell, K. Yang, J. Deng
'''

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, conv2d, max_pool2d, dropout
from tensorflow.image import resize_nearest_neighbor


#### Base blocks ####


def residual(input, n_feats_1=128, n_output_feats=256, name='Residual'):
    '''
    Builds residual module of size :
    1*1*n_feats_1 -> 3*3*n_feats_1 ->  1*1*n_output_feats

    For computational purposes each layer is preceded by a batch_norm

    :param input: (tensor NHWC)
    :param n_feats_1: (int) the number of features to output from the first layer
    :param n_output_feats: (int) number of output features
    :param name: name to be given to the block for tensor-board

    :return: (tensor NHWC) [N_input, H_input, W_input, n_output_feats]
    '''

    with tf.name_scope(name):
        norm_1 = batch_norm(input)
        conv_1 = conv2d(norm_1, n_feats_1, kernel_size=1, name='Convolution_1')
        norm_2 = batch_norm(conv_1)
        conv_2 = conv2d(norm_2, n_feats_1, kernel_size=3, name='Convolution_2')
        norm_3 = batch_norm(conv_2)
        out = conv2d(norm_3, n_output_feats, kernel_size=1, name='Out')

        return out

def skip_layer(input, n_output_feats=256, name='Skip_Layer'):
    '''
    Makes a skip layer that reshapes the number of features of the input of a residual block if needed
    so it can be summed with the output of the residual

    :param input: (tensor NHWC)
    :param n_output_feats: (int) number of output features
    :param name: name to be given to the block for tensor-board

    :return: (tensor NHWC) [N_input, H_input, W_input, n_output_feats]
    '''

    with tf.name_scope(name):
        if input.get_shape().as_list()[3] == n_output_feats:

            return input

        else:
            conv = conv2d(input, n_output_feats, kernel_size=1, name='Convolution')

            return conv


def residual_block(input, n_feats_1=128, n_output_feats=256, name='Residual_Block'):
    '''
    Builds residual block

    :param input: (tensor NHWC)
    :param n_feats_1: (int) the number of features to output from the first layer of the residual
    :param n_output_feats: (int) number of output features
    :param name: name to be given to the block for tensor-board

    :return: (tensor NHWC) [N_input, H_input, W_input, n_output_feats]
    '''

    with tf.name_scope(name):
        res = residual(input, n_feats_1=n_feats_1, n_output_feats=n_output_feats)
        skip = skip_layer(input, n_output_feats=n_output_feats)
        out = tf.add_n([res, skip], name='Addition_Layer')
        
        return out


#### Building blocks ####

def starting_block(input, n_feats=256, name='Starting_Block'):
    '''
    Builds the first block of the network

    :param input: (tensor NHWC)
    :param n_feats: (int) number of features to output throughout the hourglass
    :param name: name to be given to the block for tensor-board

    :return: (tensor NHWC) [N_input, H_input/4, W_input/4, n_feats]
    '''

    with tf.name_scope(name):
        norm_1 = batch_norm(input)
        conv = conv2d(norm_1, n_feats, kernel_size=7, stride=2, name='Convolution')
        res_down = residual_block(conv, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Down_Residual_Block')
        pooled = max_pool2d(res_down, pool_size=4, stride=4, name='Pooling_Layer')
        res = residual_block(pooled, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Residual_Block_1')
        out = residual_block(res, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Residual_Block_2')

        return out


def hourglass(input, num_max_pools=3, n_feats=256, name='Hourglass'):
    '''
    Builds an hourglass

    :param input: (tensor NHWC)
    :param num_max_pools: (int) number of max pooling layers - 1
    :param n_feats: (int) number of features to output throughout the hourglass
    :param name: name to be given to the block for tensor-board

    :return: (tensor NHWC) [N_input, H_input, W_input, n_feats]
    '''

    with tf.name_scope(name):
        res_down = residual_block(input, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Down_Residual_Block')
        branch_off = residual_block(res_down, n_feats_1=int(n_feats / 2), n_output_feats=n_feats,
                                    name='Residual_Block_off_branch')
        pooled = max_pool2d(res_down, pool_size=2, stride=2, name='Pooling_Layer')
        if num_max_pools > 0:
            main_out = hourglass(pooled, num_max_pools=num_max_pools - 1, n_feats=256, name='Sub_Hourglass')

        else:
            main_out = residual_block(pooled, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Smaller_Residual_Block')

        uped = resize_nearest_neighbor(main_out, tf.shape(main_out)[1:3] * 2, name= 'Up_Sampling_Layer')
        res_up = residual_block(uped, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Up_Residual_Block')
        out = tf.add_n([branch_off, res_up], name='Addition_Layer')

        return out


def post_hourglass_block(input, n_feats=256, output_dim=13, name='Post_Hourglass_Block', training=True):
    '''
    Builds a post hourglass module with two branches :
        - two consecutive residual nodes
        - a branch out from the first residual node that computes heatmaps then are remaped to fit the n_feats

    :param input: (tensor NHWC)
    :param n_feats: (int) number of features to output throughout the hourglass
    :param output_dim: (int) number of joints to compute
    :param name: name to be given to the block for tensor-board
    :param training: (Boolean) if True the dropout layer drops 10% of the weights and connections

    :return: (tensor) (tensor)
    '''

    with tf.name_scope(name):
        norm = batch_norm(input, activation_fn=tf.nn.relu)
        res_1 = residual_block(norm, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Residual_Block_1')
        res_2 = residual_block(res_1, n_feats_1=int(n_feats / 2), n_output_feats=n_feats, name='Residual_Block_2')
        dropped = dropout(res_2, rate=0.1, training=training, name='Dropout_Layer')
        heat_map = conv2d(res_1, output_dim, kernel_size=1, name='Compute_Heatmap')
        remaped = conv2d(heat_map, n_feats, kernel_size=1, name='Remap_Heatmap')
        out = tf.add_n([dropped, remaped], name='Addition_Layer')

        return out, heat_map


def output_block(input, n_feats=256, output_dim=13, name='Output_Block'):
    '''
    Computes the output of the network

    :param input: (tensor NHWC)
    :param n_feats: (int) number of features to output throughout the hourglass
    :param output_dim: (int) number of joints to compute
    :param name: name to be given to the block for tensor-board

    :return: output_dim heatmaps
    '''

    with tf.name_scope(name):
        norm = batch_norm(input, activation_fn=tf.nn.relu)
        conv = conv2d(norm, n_feats, kernel_size=1, name='convolution')
        heat_map = conv2d(conv, output_dim, kernel_size=1, name='Compute_Heatmap')

        return heat_map



