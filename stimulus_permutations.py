import os
import numpy as np
import pickle
import tensorflow as tf
import copy


class Stimulus:

    def __init__(self, config):

        self.config         = config
        self.n_outputs      = 10


        self.perms = [np.random.choice(10, size = (10), replace = False) for _ in range(1000)]

        """
        self.perms = [np.concatenate((np.random.choice([0,1], size = (2), replace = False), \
            np.random.choice([2,3], size = (2), replace = False), \
            np.random.choice([4,5], size = (2), replace = False), \
            np.random.choice([6,7], size = (2), replace = False), \
            np.random.choice([8,9], size = (2), replace = False))) for _ in range(1000)]
        """


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
        self.mnist_inds = [np.where(self.mnist_labels==i)[0] for i in range(10)]


    def generate_batch(self, possible_contexts):
        """ Generate a batch of randomly permuted MNIST images, based
            on the current task. """

        # Pick out batch data and labels
        batch_data   = np.zeros((self.config['trials_per_sequence'],
                                 self.config['batch_size'],
                                 28**2), dtype=np.float32)
        batch_labels = np.zeros((self.config['trials_per_sequence'],
                                 self.config['batch_size'],
                                 self.n_outputs), dtype=np.float32)
        batch_context = np.zeros((self.config['batch_size'],
                               self.config['n_output_rotations']), dtype=np.float32)

        # determine how output labels are rotated for each sequence
        context = np.random.choice(possible_contexts,
                                   size = self.config['batch_size'])

        for n in range(self.config['batch_size']):
            batch_context[n, context[n]] = 1.

        for n in range(self.config['trials_per_sequence']):


            #ind = np.random.choice(self.n_images, size = self.config['batch_size'])
            #ind_noise = np.random.choice(self.n_images, size = self.config['batch_size'])
            #start_ind = np.random.choice(28, size = self.config['batch_size'])
            #start_ind = task_rotation * 5


            #batch_data[n, :, :] = np.reshape(self.mnist_images[ind, ...], (self.config['batch_size'], -1))
            for j in range(self.config['batch_size']):

                ind = np.random.choice(self.n_images)
                label = self.mnist_labels[ind]


                s = copy.copy(self.mnist_images[ind, ...])
                #s_noise = copy.copy(self.mnist_images[ind_noise, ...])
                #if context[j] >= 80:
                #s += 0.75 * s_noise
                #q = np.random.choice(28)
                #q = (start_ind[j] + n) % 28
                #q = start_ind[j]  % 28
                #s = np.concatenate((s[q:, :], s[:q, :]), axis = 0)
                #q = np.random.choice(28)
                #s = np.concatenate((s[:, q:], s[:, :q]), axis = 1)
                #s += np.random.normal(0, 0.25, size = s.shape)
                #s /= np.sum(s)
                batch_data[n, j, :] = np.reshape(s, (-1))

                label_ind = self.perms[context[j]][label]
                batch_labels[n, j, label_ind] = 1

                #batch_data[n, j, :] = batch_data[n, j, self.perms[context_inp[j]]]



        # Return images and labels

        return batch_data, batch_labels, batch_context
