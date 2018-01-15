'''
This script creats data set to be fed into the network
'''

from scipy.io import loadmat
from scipy.misc import imsave
from imageio import get_reader
import numpy as np
import os

def Create_data_set(set, rate):
    '''
    Create a working data set for the model.
    Images are saved as .jpg and joints matrix as .npy
    The data is saved in Data/'set'/...

    :param set: 'train', 'test' or 'val'
    :param run: 'run0', 'run1' or 'run2'
    :param rate: (int) how many images to keep per videos

    :return: list.txt, a list of all images in the created data set
    '''

    data_list = open('Data/{}/list.txt'.format(set), 'w')  # list of names of the data processed into the data set
    perm = [7, 4, 1, 2, 5, 8, 10, 18, 16, 17, 19, 21, 12]

    for run in ['run0', 'run1', 'run2']:
        count=0
        path = 'Data/SURREAL/data/cmu/{0}/{1}'.format(set, run)
        try :
            print('-- Treating : {} files'.format(len(os.listdir(path))))
            for file in os.listdir(path):
                count+=1
                sub_files = np.unique([_.replace('_info.mat', '').replace('.mp4', '') for _ in os.listdir('{0}/{1}'.format(path, file))])
                for sub_file in sub_files:
                    mat = loadmat('{0}/{1}/{2}_info.mat'.format(path, file, sub_file))
                    joints = mat['joints2D']
                    to_keep = np.random.choice([True, False], size=np.shape(joints)[2], p=[rate, 1 - rate])
                    for _, ind in zip(range(np.shape(joints)[2]), to_keep):
                        if ind:
                            np.save('Data/{0}/matrix/{1}_{2}_{3}'.format(set, file, sub_file, _), joints[:, perm, _].astype(int))

                    # Images
                    vid = get_reader('{0}/{1}/{2}.mp4'.format(path, file, sub_file))
                    for _, ind in zip(range(vid.get_length()), to_keep):
                        img = vid.get_data(_)
                        if ind:
                            imsave('Data/{0}/images/{1}_{2}_{3}.jpg'.format(set, file, sub_file, _), img)

                    for _, ind in zip(range(vid.get_length()), to_keep):
                        if ind:
                            data_list.write('{0}_{1}_{2}\n'.format(file, sub_file, _))

                    if count % 10:
                        print(count/len(os.listdir(path)))
        except:
            print('No data in ' + run)

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
        #np.save('Data/LSP/matrix/{0}'.format(file_list[_]), temp_joints.astype(int))
        data_list.write('{0}\n'.format(file_list[_]))