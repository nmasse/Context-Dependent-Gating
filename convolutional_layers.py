### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import tensorflow as tf
import pickle
import time
import stimulus
from parameters import *


class ConvolutionalLayers:

    def __init__(self):

        # Reset TensorFlow graph
        tf.reset_default_graph()
        # Train on CIFAR-10 task
        task_id = 0
        self.batch_size = 1024

        current_layers = np.array(par['layer_dims'])
        current_task = str(par['task'])
        current_batch_size = int(par['batch_size'])

        if par['task'] == 'cifar':
            par['layer_dims'][-1] = 10
            print('Training convolutional layers on the CIFAR-10 dataset...')
        elif par['task'] == 'imagenet':
            par['layer_dims'][-1] = 100
            print('Training convolutional layers on the CIFAR-10 and CIFAR-100 datasets...')
        par['task']  = 'cifar' # convolutional layers always trained on cifar
        par['batch_size'] = 1024
        # Create placeholders for the model
        input_data  = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'stim')
        target_data  = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'target')
        mask   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'mask')

        with tf.Session() as sess:
            cifar_model   = self.model(input_data, target_data, mask)
            sess.run(tf.global_variables_initializer())
            t_start = time.time()

            s = stimulus.Stimulus(include_cifar10 = True, labels_per_task = par['layer_dims'][-1], include_all = True)

            for i in range(par['n_train_batches_conv']):

                x, y, m = s.make_batch(task_id, test = False)
                _, loss  = sess.run([self.train_op, self.loss], feed_dict = {input_data:x, target_data: y, mask:m})

                if i%1000 == 0:
                    print('Iteration ', i, ' Loss ', loss)

            W = {}
            for var in tf.trainable_variables():
                W[var.op.name] = var.eval()
            fn = par['save_dir'] + current_task + '_conv_weights.pkl'
            pickle.dump(W, open(fn,'wb'))
            print('Convolutional weights saved in ', fn)

            # Revert to old parameters
            update_parameters({'task': current_task, 'layer_dims': current_layers, 'batch_size': current_batch_size})


    def model(self, input_data, target_data, mask):

        conv1 = tf.layers.conv2d(inputs=input_data, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

        # Now 64x64x32
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
        conv2 = tf.nn.dropout(conv2, par['conv_drop_keep_pct'])

        # Now 16x16x32
        conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)

        # Now 16x16x64
        conv4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding='same')
        conv4 = tf.nn.dropout(conv4, par['conv_drop_keep_pct'])

        self.x = tf.reshape(conv4,[par['batch_size'], -1])

        for n in range(par['n_layers']-1):
            scope_name = 'layer' + str(n)
            with tf.variable_scope(scope_name):

                W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                b = tf.get_variable('b', initializer = tf.zeros([1,par['layer_dims'][n+1]]), trainable = True)

                if n < par['n_layers']-2:
                    self.x = tf.nn.dropout(tf.nn.relu(tf.matmul(self.x, W) + b), par['drop_keep_pct'])
                else:
                    self.y = tf.matmul(self.x,W) + b  - (1-mask)*1e16

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, labels = target_data, dim=1))

        optimizer = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])

        self.train_op = optimizer.minimize(self.loss)
