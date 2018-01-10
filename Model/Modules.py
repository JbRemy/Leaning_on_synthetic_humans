'''
Implementation of a stacked hourglass network as described in
"Stacked Hourglass Networks for Human Pose Estimation"  A. Newell, K. Yang, J. Deng
'''

import tensorflow as tf

def make_Residual_Module():
    '''
        Builds residual module
        128*1*1 -> 128*3*3 -> 256*1*1
    '''
    def __init__(self, n_input_feats, n_ouput_feats):