
import pickle
import yaml
import os, sys
import numpy as np
from itertools import product
from stimulus import Stimulus
import tensorflow.compat.v1 as tf
from model import Model
import matplotlib.pyplot as plt
tf.disable_v2_behavior()


def main():

    config = read_config('config.yaml')

    # Define all placeholders
    input_pl, target_pl, lr_pl = generate_placeholders(config)

    # Set up stimulus and accuracy recording
    stim = Stimulus(config)

    # Start Tensorflow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/gpu:0' if config['use_gpu'] else '/cpu:0' 
        with tf.device(device):
            # Check order against args unpacking in model if editing
            model = Model(input_pl, target_pl, lr_pl, config)

        # Initialize variables and start the timer
        sess.run(tf.global_variables_initializer())

        for i in range(config['n_iterations']):

            # linear ramp for learning rate
            learning_rate = config['learning_rate'] * np.minimum(i/2000, 1.)

            # Generate a batch of stimulus data for training
            batch_data, batch_labels = stim.generate_batch()

            # Put together the feed dictionary
            feed_dict = {input_pl:batch_data,
                        target_pl:batch_labels,
                        lr_pl: learning_rate}

            # Calculate and apply gradients
            _, rewards, activity = sess.run([model.train_op,
                                            model.reward,
                                            model.activity],
                                            feed_dict = feed_dict)

            # Record accuracies
            mean_activity = np.mean(np.stack(activity))
            rewards = np.stack(rewards)
            accuracy = np.mean(rewards > 0, axis = (1,2))


            # Display network performance
            if i%100 == 0:
                print('Iteration ', i, 'activity', mean_activity)
                print('Accuracy ', accuracy)
                print()


def generate_placeholders(config):


    input_pl = tf.placeholder(tf.float32, shape=[config['trials_per_sequence'],
                                                 config['batch_size'],
                                                 config['n_input']])  # input data

    target_pl = tf.placeholder(tf.float32, shape=[config['trials_per_sequence'],
                                                 config['batch_size'],
                                                 config['n_output']])  # input data

    lr_pl  = tf.placeholder(tf.float32, shape=[]) # learning rate

    return input_pl, target_pl, lr_pl


def read_config(config_filename):

    with open(config_filename, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config


if __name__ == '__main__':

    main()
