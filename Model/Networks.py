'''
Class for a Stacked Hourglass Network
'''

import numpy as np
import tensorflow as tf
import time

from Model.Modules import hourglass, starting_block, post_hourglass_block, output_block
from Model.Utils import make_batch, compute_loss, mat_to_joints

class Stacked_Hourglass():
    def __init__(self, n_stacks=8, num_max_pools=3, n_feats=256, name='Stacked_Hourglass'):
        '''
        Initializes the parameters of the network

        :param n_stacks: (int) number of hourglass
        :param num_max_pools: (int) number of max pooling layers - 1
        :param n_feats: (int) number of features to output throughout the hourglass
        '''

        self.name = name

        # Parameters of the network
        self.n_stacks = n_stacks
        self.num_max_pools = num_max_pools
        self.n_feats = n_feats


    def fit(self, X_list, n_epochs, input_H=256, input_W=256, batch_size=16, output_dim=13, learning_rate=10e-4,
            print_every_epoch=100, save_every_epoch=100, persistent_save=False, save_path=""):
        '''
        trains the network

        :param X_list: (.txt file) names of the training images
        :param n_epochs: (int) number of epochs
        :param input_H: (int) Height of input images
        :param input_W: (int) Width of input images
        :param batch_size: (int) batch size
        :param output_dim: (int) number of joints to compute
        :param learning_rate: (float)
        :param print_every_epoch: (int) when to output results to the oerator
        :param save_every_epoch: (int) when to save a point in the learning curve
        :param persistent_save: (Boolean) if True the model is saved to 'save_path'
        :param save_path: (string) where to save the model
        '''

        with open(X_list, 'r') as file:
            lines = file.readlines()
            length = len(lines)

        self.input_H = input_H
        self.input_W = input_W
        self.output_dim = output_dim

        start_time = time.time()
        with tf.device('/cpu:0'):
            print('- Initializing network')
            with tf.name_scope('inputs'):
                X = tf.placeholder(tf.float32, [None, input_H, input_W, 3], name='X_train')
                y = tf.placeholder(tf.float32, [None, int(input_H/4), int(input_W/4), output_dim], name='y_train')

            network = self._build_network(X, output_dim=output_dim)

            with tf.name_scope('Loss'):

                loss = compute_loss(network, y)
                tf.summary.scalar('loss', loss, collections=['train'])
                merged_summary = tf.summary.merge_all('train')

            if persistent_save:
                summary_train = tf.summary.FileWriter('{}/logs/train/'.format(save_path), tf.get_default_graph())

            with tf.name_scope('Optimizer'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
                minimizer = optimizer.minimize(loss)

            print('|-- done ({})'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
            print('- Starting training')
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                for epoch in range(n_epochs):
                    avg_cost = 0
                    for batch_number in range(int(length/batch_size)-1):
                        X_batch, y_batch = make_batch(X_list, batch_size)
                        _, c = sess.run([minimizer, loss], feed_dict={X: X_batch, y: y_batch})
                        avg_cost += c / (int(length/batch_size) * batch_size)

                    X_batch, y_batch = make_batch(X_list, batch_size, input_H, input_W, output_dim)
                    _, c, summary = sess.run([minimizer, loss, merged_summary], feed_dict={X: X_batch, y: y_batch})
                    avg_cost += c / (int(n_epochs / batch_size) * batch_size)
                    if epoch % save_every_epoch:
                        summary_train.add_summary(summary, epoch)
                        summary_train.flush()

                    if epoch % print_every_epoch:
                        print('|-- epoch {0} done ({1}) :'.format(epoch, time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
                        print('| |-- avg_cost = {}'.format(avg_cost))

                print('- Training done ({})'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
                print('|-- Learning curve saved to Data/logs/train/')
                saver.save(sess, '/tmp/model.ckpt')
                if persistent_save:
                    saver.save(sess, save_path)
                    print('- Model saved to {}'.format(save_path))

            summary_train.close()
            sess.close()


    def predicit(self, X_list, batch_size, set='test', trainning=False, path= 'tmp/model/ckpt'):
        '''
        Computes output of a pre-trained model over a test batch

        :param X_list: (.txt file) names of the test images
        :param batch_size: (int) batch size
        :param set: (str) 'train' 'test' 'val'
        :param trainning: (boolean) False
        :param path: (str) path to the saved model

        :return: (np array) set of predicted joints for all images in the batch, (np array) true joints, (float) loss over the batch
        '''

        X_batch, y_batch = make_batch(X_list, batch_size, set=set, trainning=trainning)
        X = tf.placeholder(tf.float32, [None, self.input_H, self.input_W, 3], name='X_train')
        y = tf.placeholder(tf.float32, [None, int(self.input_H / 4), int(self.input_W / 4), self.output_dim], name='y_train')
        network_output = tf.get_variable('Network/Output', [self.n_stacks, None, int(self.input_H/4), int(self.input_W/4),
                                                            self.output_dim], initializer = tf.zeros_initializer)
        loss = tf.get_variable('Loss/loss', [1], initializer=tf.zeros_initializer)
        saver = tf.train.Saver({'Network/Output' : network_output, 'Loss/loss': loss})
        with tf.Session() as sess:
            saver.restore(sess, path)
            heat_maps, loss = sess.run([network_output, loss], feed_dict={X: X_batch, y: y_batch})

        heat_maps = heat_maps.eval()
        joints = np.zeros([batch_size, 2, self.output_dim])
        for _ in range(batch_size):
            joints = mat_to_joints(heat_maps[self.n_stacks, _, :, :, :])

        sess.close()

        return joints, y_batch, loss


    def _build_network(self, input, output_dim=13):
        '''
        Builds the full network

        :param input: (tensor NHWC)
        :param output_dim: (int) number of joints to compute

        :return: (tensor BNHWC)
        '''

        with tf.name_scope('Network'):
            start = starting_block(input, n_feats=self.n_feats)
            stacks_out = start
            heat_maps = []
            for _ in range(1, self.n_stacks):
                hg = hourglass(stacks_out, num_max_pools=self.num_max_pools, n_feats=self.n_feats,
                               name='Hourglass_{}'.format(_))
                post_hg, heat_map = post_hourglass_block(hg, n_feats=self.n_feats, output_dim=output_dim,
                                                             name='Post_Hourglass_Module_{}'.format(_))
                heat_maps.append(heat_map)
                stacks_out = tf.add_n([post_hg, stacks_out])

            heat_maps.append(output_block(stacks_out, n_feats=self.n_feats, output_dim=output_dim))
            out = tf.stack(heat_maps, name='Output')

            return out


