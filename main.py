'''
Main file for testing Adversarial Learning on the SURREAL data set
'''

from Data.Create_data_set import Create_data_set
from Model.Networks import Stacked_Hourglass

Create_data_set('train', 'run1')

Neural_Network = Stacked_Hourglass(n_stacks=2)
Neural_Network.fit('Data/train/list.txt', n_epochs=7, save_path='Model/test_model', persistent_save=True)

X_batch, joints, y_batch, c = Neural_Network.predicit('Data/LSP/list.txt', batch_size=5, set='LSP', path='to_extract/Model_2')


from Model.Networks import Stacked_Hourglass
Neural_Network = Stacked_Hourglass(n_stacks=1)
Neural_Network.input_H=256
Neural_Network.input_W=256
Neural_Network.output_dim=13
Neural_Network.batch_size=6
model = 'Model_Adversarial_final'
RMSE, PDJ = Neural_Network.eval('Data/LSP/list.txt', set='LSP', path= 'to_extract/'+model, Source=False)

from Model.Networks import Stacked_Hourglass
import numpy as np
Neural_Network = Stacked_Hourglass(n_stacks=1)
Neural_Network.input_H=256
Neural_Network.input_W=256
Neural_Network.output_dim=13
Neural_Network.batch_size=6
model = 'Model_Advers_full_train'
X_batch, joints, true_joints = Neural_Network.predict('Data/LSP/test_list.txt', batch_size=5 , set='LSP', path= 'to_extract/'+model, adversarial=True, Source=False)
np.save('to_extract/'+model+'/X_batch_test', X_batch)
np.save('to_extract/'+model+'/joints_test', joints)
np.save('to_extract/'+model+'/true_joints_test', true_joints)