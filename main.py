'''
Main file for testing Adversarial Learning on the SURREAL data set
'''

from Data.Create_data_set import Create_data_set
from Model.Networks import Stacked_Hourglass

Create_data_set('train', 'run1')

Neural_Network = Stacked_Hourglass(n_stacks=2)
Neural_Network.fit('Data/train/list.txt', n_epochs=7, save_path='Model/test_model', persistent_save=True)

X_batch, joints, y_batch, c = Neural_Network.predicit('Data/LSP/list.txt', batch_size=5, set='LSP', path='to_extract/Model_2')



