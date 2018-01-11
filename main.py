'''
Main file for testing Adversarial Learning on the SURREAL data set
'''

from Data.Create_data_set import Create_data_set

# Creating working data_sets
for data_set, run in ['train', 'test', 'val']:
    for run_ in ['run0', 'run1', 'run2']:
        Create_data_set(data_set, run_)

Create_data_set('val', 'run0')