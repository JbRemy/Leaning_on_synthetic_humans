'''
Main file for testing Adversarial Learning on the SURREAL data set
'''

#from Data.Create_data_set import Create_data_set
from Model.Networks import Stacked_Hourglass

# Creating working data_sets
# for data_set, run in ['train', 'test', 'val']:
#   for run_ in ['run0', 'run1', 'run2']:
#        Create_data_set(data_set, run_)

#Create_data_set('train', 'run1')

Neural_Network = Stacked_Hourglass(n_stacks=2)
Neural_Network.fit('Data/train/list.txt', n_epochs= 1,save_path='Model/test_model')