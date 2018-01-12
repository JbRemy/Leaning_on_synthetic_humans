'''
Utilities for the networks
'''

import numpy as np
from numpy.random import choice
import tensorflow as tf


def make_batch(X_list, batch_size):


    with open(X_list, 'r') as file:
        lines = choice(file.readlines(), size=batch_size)

     return 0, 0