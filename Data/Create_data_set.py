'''
This script creats data set to be fed into the network
'''

from scipy.io import loadmat
from scipy.misc import imsave
from imageio import get_reader
import numpy as np
import os

def Create_data_set(set, run):
    '''
    Create a working data set for the model.
    Images are saved as .jpg and joints matrix as .npy
    The data is saved in Data/'set'/...

    :param set: 'train', 'test' or 'val'
    :param run: 'run0', 'run1' or 'run2'

    :return: list.txt, a list of all images in the created data set
    '''

    perm = [7, 4, 1, 2, 5, 8, 10, 18, 16, 17, 19, 21, 12]
    path = 'Data/SURREAL/data/cmu/{0}/{1}'.format(set, run)
    data_list = open('Data/{}/list.txt'.format(set), 'w') # list of names of the data processed into the data set

    for file in os.listdir(path):
        mat_files = [_ for _ in os.listdir('{0}/{1}'.format(path, file)) if 'mat' in _]
        vid_files = [_ for _ in os.listdir('{0}/{1}'.format(path, file)) if 'mp4' in _]

        # Joints matrix
        for sub_file in mat_files:
            mat = loadmat('{0}/{1}/{2}'.format(path, file, sub_file))
            joints = mat['joints2D']
            for _ in range(np.shape(joints)[2]):
                np.save('Data/{0}/matrix/{1}_{2}'.format(set, file, _), joints[:, perm, _].astype(int))

        # Images
        for sub_file in vid_files:
            vid = get_reader('{0}/{1}/{2}'.format(path, file, sub_file))
            for _ in range(vid.get_length()):
                img = vid.get_data(_)
                imsave('Data/{0}/images/{1}_{2}.jpg'.format(set, file, _), img)

        for _ in range(vid.get_length()):
            data_list.write('{0}_{1}\n'.format(file, _))

    data_list.close()

def Create_LSP():
    '''
    Builds the LSP data for validation
    '''

    data_list = open('Data/LSP/list.txt'.format(set), 'w')

    file_list = [_.replace('.jpg','') for _ in os.listdir('Data/LSP/images')]

    mat =  loadmat('Data/LSP/joints')
    joints = mat['joints']
    for _ in range(2000):
        temp_joints = joints[0:2,0:13,_]
        np.save('Data/LSP/matrix/{0}'.format(file_list[_]), temp_joints.astype(int))
        data_list.write('{0}_{1}\n'.format(file_list[_], _))