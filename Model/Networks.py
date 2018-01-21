'''
Class for a Stacked Hourglass Network
'''

import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib.layers import flatten

from numpy.random import choice

from Model.Modules import hourglass, starting_block, post_hourglass_block, output_block, domain_classification_output
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


    def fit(self, X_list, n_epochs, input_H=256, input_W=256, batch_size=6, output_dim=13, learning_rate=10e-4,
            print_every_batch=100, save_every_batch=100, persistent_save=False, save_path="", preproc=False):
        '''
        trains the network

        :param X_list: (.txt file) names of the training images
        :param n_epochs: (int) number of epochs
        :param input_H: (int) Height of input images
        :param input_W: (int) Width of input images
        :param batch_size: (int) batch size
        :param output_dim: (int) number of joints to compute
        :param learning_rate: (float)
        :param print_every_batch: (int) when to output results to the oerator
        :param save_every_batch: (int) when to save a point in the learning curve
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
        with tf.device('/GPU:0'):
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
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = learning_rate
                learning_rate_decay = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                           2000, 0.5, staircase=True, name='Learning_Rate')
                optimizer = tf.train.RMSPropOptimizer(learning_rate_decay)
                minimizer = optimizer.minimize(loss)

            print('|-- done ({})'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
            print('- Starting training')
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                sess.run(init)
                avg_cost = 0
                for epoch in range(n_epochs):
                    for batch_number in range(int(length/batch_size)):
                        try :
                            X_batch, y_batch = make_batch(X_list, batch_size, preproc=preproc)
                            _, c, summary = sess.run([minimizer, loss, merged_summary], feed_dict={X: X_batch, y: y_batch})
                            avg_cost = avg_cost*(batch_number + epoch*int(length/batch_size)) / (batch_number + 1 + epoch*int(length/batch_size)) + c / (batch_number + 1 + epoch*int(length/batch_size))
                            if batch_number % save_every_batch == 0:
                                summary_train.add_summary(summary, batch_number + epoch * int(length/batch_size))
                                summary_train.flush()

                            if batch_number % print_every_batch == 0:
                                print('|-- Epoch {0} Batch {1} done ({2}) :'.format(epoch, batch_number, time.strftime("%H:%M:%S", time.gmtime(
                                    time.time() - start_time))))
                                print('| |-- avg_cost = {}'.format(avg_cost))

                        except:
                            print('exeption')

                print('- Training done ({})'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
                print('|-- Learning curve saved to Data/logs/train/')
                saver.save(sess, '/tmp/model.ckpt')
                if persistent_save:
                    saver.save(sess, '{}/model.ckpt'.format(save_path))
                    print('- Model saved to {}'.format(save_path))

                sess.close()

            summary_train.close()



    def adversarial_fit(self, Source_list, Target_list, n_epochs, input_H=256, input_W=256, batch_size=6, output_dim=13, learning_rate=10e-4,
            print_every_batch=100, save_every_batch=100, persistent_save=False, save_path=""):
        '''
        trains the network with domain adversarial learning

        :param Source_list: (.txt file) names of the source training images
        :param Target_list: (.txt file) names of the target training images
        :param n_epochs: (int) number of epochs
        :param input_H: (int) Height of input images
        :param input_W: (int) Width of input images
        :param batch_size: (int) batch size
        :param output_dim: (int) number of joints to compute
        :param learning_rate: (float)
        :param print_every_batch: (int) when to output results to the oerator
        :param save_every_batch: (int) when to save a point in the learning curve
        :param persistent_save: (Boolean) if True the model is saved to 'save_path'
        :param save_path: (string) where to save the model
        '''

        start_time = time.time()

        with open(Source_list, 'r') as file:
            lines = file.readlines()
            length = len(lines)

        self.batch_size = batch_size
        self.input_H = input_H
        self.input_W = input_W
        self.output_dim = output_dim

        with tf.device('/GPU:0'):
            print('- Initializing network')
            with tf.name_scope('inputs'):
                X = tf.placeholder(tf.float32, [None, input_H, input_W, 3], name='X_train')
                y = tf.placeholder(tf.float32, [None, int(input_H/4), int(input_W/4), output_dim], name='y_train')
                y_domain = tf.placeholder(tf.float32, [None, 2], name='y_domain')

            network, domain_out = self._build_network(X, output_dim=output_dim, adversarial=True)

            with tf.name_scope('Loss'):
                loss = compute_loss(network, y)
                tf.summary.scalar('loss', loss, collections=['train'])
                loss_domain = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=domain_out, labels=y_domain), name='loss_domain')
                tf.summary.scalar('loss_domain', loss_domain, collections=['train'])
                joint_loss = tf.add_n([loss, loss_domain])
                tf.summary.scalar('joint_loss', joint_loss, collections=['train'])
                merged_summary = tf.summary.merge_all('train')

            if persistent_save:
                summary_train = tf.summary.FileWriter('{}/logs/train/'.format(save_path), tf.get_default_graph())

            with tf.name_scope('Optimizer'):
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = learning_rate
                learning_rate_decay = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                 3000, 0.9, staircase=True, name='Learning_Rate')
                optimizer_Source = tf.train.RMSPropOptimizer(learning_rate_decay)
                minimizer_Source = optimizer_Source.minimize(joint_loss)

                optimizer_Target = tf.train.RMSPropOptimizer(learning_rate_decay)
                minimizer_Target = optimizer_Target.minimize(loss_domain)


            print('|-- done ({})'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
            print('- Starting training')

            minimizer_list = [minimizer_Source, minimizer_Target]
            X_list = [Source_list, Target_list]
            set_list = ['train', 'LSP']

            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                sess.run(init)
                for epoch in range(n_epochs):
                    for batch_number in range(int(length/batch_size)):
                        batch_type = choice([0, 1], p = [0.7, 0.3])
                        passed = 0
                        while passed == 0:
                            try :
                                X_batch, y_batch, y_domain_batch = make_batch(X_list[batch_type], batch_size, set=set_list[batch_type], adversarial=True, Source=bool(batch_type==0))
                                passed = 1

                            except:
                                print('exception')

                        _, summary = sess.run([minimizer_list[batch_type], merged_summary], feed_dict={X: X_batch, y: y_batch, y_domain: y_domain_batch})
                        if batch_number % save_every_batch == 0:
                            summary_train.add_summary(summary, batch_number + epoch * int(length/batch_size))
                            summary_train.flush()

                        if batch_number % print_every_batch == 0:
                            print('|-- Epoch {0} Batch {1} done ({2}) :'.format(epoch, batch_number, time.strftime("%H:%M:%S", time.gmtime(
                            time.time() - start_time))))

                print('- Training done ({})'.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))))
                print('|-- Learning curve saved to {}/logs/train/'.format(save_path))
                saver.save(sess, '/tmp/model.ckpt')
                if persistent_save:
                    saver.save(sess, '{}/model.ckpt'.format(save_path))
                    print('- Model saved to {}'.format(save_path))

            summary_train.close()
            sess.close()


    def predict(self, X_list, batch_size, set='LSP', path= 'tmp/model/ckpt', adversarial=False, Source=False):
        '''
        Computes output of a pre-trained model over a test batch

        :param X_list: (.txt file) names of the test images
        :param batch_size: (int) batch size
        :param set: (str) 'train' 'test' 'val'
        :param trainning: (boolean) False
        :param path: (str) path to the saved model

        :return: (np array) set of predicted joints for all images in the batch, (np array) true joints, (float) loss over the batch
        '''

        with tf.device('/GPU:0'):
            if adversarial :
                X_batch, y_batch, y_domain = make_batch(X_list, batch_size, set=set, Source=Source, adversarial=adversarial)

            else:
                X_batch, y_batch = make_batch(X_list, batch_size, set=set, Source=Source, adversarial=adversarial)

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                saver = tf.train.import_meta_graph(path+'/model.ckpt.meta')
                saver.restore(sess, tf.train.latest_checkpoint(path))
                graph = tf.get_default_graph()
                X = graph.get_tensor_by_name("inputs/X_train:0")
                y = graph.get_tensor_by_name("inputs/y_train:0")
                network = graph.get_tensor_by_name('Network/Output:0')
                heat_maps = sess.run(network, feed_dict={X: X_batch, y: y_batch})

            out_heat_maps = np.zeros([self.batch_size, 64, 64, 13])
            for _ in range(self.n_stacks):
                out_heat_maps = np.add(out_heat_maps, heat_maps[_, :, :, :, :])

            joints = np.zeros([batch_size, 2, self.output_dim])
            for _ in range(batch_size):
                joints[_, :, :] = mat_to_joints(out_heat_maps[_, :, :, :])

            true_joints = np.zeros([batch_size, 2, self.output_dim])
            for _ in range(y_batch.shape[0]):
                true_joints[_, :, :] = mat_to_joints(y_batch[_, :, :, :])
            sess.close()

            return X_batch, joints, true_joints

    def eval(self, X_list, set='LSP', path= 'tmp/model/ckpt', Source=True):
        '''
        Compute RMSE and Elbows PDJ loss on the testing data set
        :param X_list:
        :param set:
        :param path:
        :param Source:
        :return:
        '''
        start_time = time.time()
        with tf.device('/GPU:0'):
            with open(X_list, 'r') as file:
                lines = file.readlines()

            length = len(lines)
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                saver = tf.train.import_meta_graph(path+'/model.ckpt.meta')
                saver.restore(sess, tf.train.latest_checkpoint(path))
                graph = tf.get_default_graph()
                X = graph.get_tensor_by_name("inputs/X_train:0")
                y = graph.get_tensor_by_name("inputs/y_train:0")
                network = graph.get_tensor_by_name('Network/Output:0')
                RMSE = 0
                num_except = 0
                print(int(length / self.batch_size))
                PDJ = [0, 0, 0]
                PDJ_ratio = [0.05, 0.2, 0.4]
                for batch_number in range(int(3 * length / self.batch_size)):
                    if batch_number % 100 == 0:
                        print('|-- Batch {0} done ({1}) :'.format(batch_number, time.strftime("%H:%M:%S", time.gmtime(
                                                                            time.time() - start_time))))
                    try :
                        X_batch, y_batch = make_batch(X_list, self.batch_size, set=set, Source=Source)
                        heat_maps = sess.run(network, feed_dict={X: X_batch, y: y_batch})

                        out_heat_maps = np.zeros([self.batch_size, 64, 64, 13])
                        for _ in range(self.n_stacks):
                            out_heat_maps = np.add(out_heat_maps, heat_maps[_, :, :, :, :])

                        joints = np.zeros([self.batch_size, 2, self.output_dim])
                        for _ in range(self.batch_size):
                            joints[_, :, :] = mat_to_joints(out_heat_maps[_, :, :, :])

                        true_joints = np.zeros([self.batch_size, 2, self.output_dim])
                        for _ in range(y_batch.shape[0]):
                            true_joints[_, :, :] = mat_to_joints(y_batch[_, :, :, :])

                        for _ in range(self.batch_size):
                            torso_width = ((true_joints[_, :, 8] - true_joints[_, :, 9])**2).sum()
                            for _1 in [7, 10]:
                                for _2 in range(3):
                                    if ((joints[_, :, _1] - true_joints[_, :, _1])**2).sum() < torso_width*PDJ_ratio[_2]**2:
                                        PDJ[_2] = PDJ[_2] + 1

                        for _ in range(self.batch_size):
                            SE = ((joints[_, :, :]  - true_joints[_, :, :] )**2).sum()
                            RMSE += SE

                    except:
                        print('exeption')
                        num_except +=1


                sess.close()

            RMSE = RMSE/((3 * int(length / self.batch_size)-num_except) * self.output_dim * 2)
            RMSE = np.sqrt(RMSE)
            PDJ = [_/((3 * int(length / self.batch_size)-num_except) * 2) for _ in PDJ]

            return RMSE, PDJ

    def _build_network(self, input, output_dim=13, adversarial=False):
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

            hg = hourglass(stacks_out, num_max_pools=self.num_max_pools, n_feats=self.n_feats,
                               name='Hourglass_{}'.format(self.n_stacks))
            out_put_heat_map = output_block(hg, n_feats=self.n_feats, output_dim=output_dim)
            heat_maps.append(out_put_heat_map)
            out = tf.stack(heat_maps, name='Output')

            if adversarial:
                domain_out = domain_classification_output(tf.add_n(heat_maps), self.batch_size)

                return out, domain_out

            else:

                return out



