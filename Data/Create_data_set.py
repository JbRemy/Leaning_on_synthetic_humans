'''
This script creats data set to be fed into the network
'''

from scipy.io import loadmat
from scipy.misc import imsave
from imageio import get_reader
import numpy as np
import os
import time
from scipy.misc import imread

from Model.lights_and_occlusions import Pre_Process_Images

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

    start_time = time.time()

    data_list = open('Data/{}/list.txt'.format(set), 'w')  # list of names of the data processed into the data set
    perm = [7, 4, 1, 2, 5, 8, 10, 18, 16, 17, 19, 21, 12]

    for run in ['run1']:#['run0', 'run1', 'run2']:
        count=0
        path = 'Data/SURREAL/data/cmu/{0}/{1}'.format(set, run)
        try :
            print('-- Treating : {} files'.format(len(os.listdir(path))))
            count += 1
            for file in os.listdir(path):
                sub_files = np.unique([_.replace('_info.mat', '').replace('.mp4', '').replace('_segm.mat', '') for _ in os.listdir('{0}/{1}'.format(path, file))])
                for sub_file in sub_files:
                    try:
                        # Info mat
                        mat = loadmat('{0}/{1}/{2}_info.mat'.format(path, file, sub_file))
                        joints = mat['joints2D']
                        to_keep = np.random.choice([True, False], size=np.shape(joints)[2], p=[rate, 1 - rate])
                        for _, ind in zip(range(np.shape(joints)[2]), to_keep):
                            if ind:
                                np.save('Data/{0}/matrix/{1}_{2}_{3}'.format(set, file, sub_file, _), joints[:, perm, _].astype(int))

                        # Segm mat
                        mat = loadmat('{0}/{1}/{2}_segm.mat'.format(path, file, sub_file))
                        to_keep = np.random.choice([True, False], size=np.shape(joints)[2], p=[rate, 1 - rate])
                        for _, ind in zip(range(np.shape(joints)[2]), to_keep):
                            if ind:
                                segm = mat['segm_{}'.format(_+1)]
                                np.save('Data/{0}/segm/{1}_{2}_{3}'.format(set, file, sub_file, _),
                                        segm.astype(int))

                        # Images
                        vid = get_reader('{0}/{1}/{2}.mp4'.format(path, file, sub_file))
                        for _, ind in zip(range(vid.get_length()), to_keep):
                            img = vid.get_data(_)
                            if ind:
                                imsave('Data/{0}/images/{1}_{2}_{3}.jpg'.format(set, file, sub_file, _), img)

                        for _, ind in zip(range(vid.get_length()), to_keep):
                            if ind:
                                data_list.write('{0}_{1}_{2}\n'.format(file, sub_file, _))

                    except:
                        print('{0}/{1} expection'.format(path, sub_file))

                print('{0} part of file done ({1})'.format(count/len(os.listdir(path)), time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))

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
        i = int(file_list[_].replace('im', ''))
        temp_joints = joints[0:2,0:13,i-1]
        np.save('Data/LSP/matrix/{0}'.format(file_list[_]), temp_joints.astype(int))
        data_list.write('{0}\n'.format(file_list[_]))

def preprocessing_images(X_list, set):
    '''
    preproceses images for the modified images model
    '''

    img_path = 'Data/{}/images'.format(set)
    segm_path = 'Data/{}/segm'.format(set)

    data_list = open('Data/{}/list_pre_processed.txt'.format(set), 'w')  # list of names of the data processed into the data set

    with open(X_list, 'r') as file:
        lines = file.readlines()

    print('There are {} to process'.format(len(lines)))
    for _ in range(len(lines)):
        img = imread('{0}/{1}.jpg'.format(img_path, lines[_].strip()))
        segm = np.load('{0}/{1}.npy'.format(segm_path, lines[_].strip()))
        try :
            img = Pre_Process_Images(img, segm)
            imsave('Data/{0}/images_pre_processed/{1}.jpg'.format(set, lines[_].strip()), img)
            data_list.write('{}\n'.format(lines[_].strip()))

        except:
            0

        if _ % 100 == 0:
            print('{} lines done'.format(_))


