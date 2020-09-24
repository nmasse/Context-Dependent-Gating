import os
import numpy as np
import pickle
import tensorflow as tf


class Stimulus:

    def __init__(self, config):

        self.config         = config
        self.n_outputs      = 10

        self.load_mnist_data()


    def load_mnist_data(self):
        """ Load MNIST data and parse its images, labels, and indices.
            Also generate the required permutations. """

        #mndata = MNIST(self.config['mnist_dir'])
        #self.mnist_images, self.mnist_labels = mndata.load_training()
        #images_fn = os.path.join(self.config['mnist_dir'], 'train-images-idx3-ubyte.gz')
        #labels_fn = os.path.join(self.config['mnist_dir'], 'train-labels-idx3-ubyte.gz')

        (self.mnist_images, self.mnist_labels), _ = tf.keras.datasets.mnist.load_data()
        self.mnist_images = np.array(self.mnist_images)/255
        self.n_images = self.mnist_images.shape[0]


    def generate_batch(self):
        """ Generate a batch of randomly permuted MNIST images, based
            on the current task. """

        # Pick out batch data and labels
        batch_data   = np.zeros((self.config['trials_per_sequence'],
                                 self.config['batch_size'],
                                 28**2), dtype=np.float32)
        batch_labels = np.zeros((self.config['trials_per_sequence'],
                                 self.config['batch_size'],
                                 self.n_outputs), dtype=np.float32)

        # determine how output labels are rotated for each sequence
        task_rotation = np.random.choice(self.config['n_output_rotations'],
                                         size = self.config['batch_size'])

        for n in range(self.config['trials_per_sequence']):

            ind = np.random.choice(self.n_images, size = self.config['batch_size'])

            batch_data[n, :, :] = np.reshape(self.mnist_images[ind, ...], (self.config['batch_size'], -1))

            # perform task label rotation
            for j in range(self.config['batch_size']):
                k = (self.mnist_labels[ind[j]] + task_rotation[j]) % self.n_outputs
                batch_labels[n, j, k] = 1

        # Return images and labels
        return batch_data, batch_labels
