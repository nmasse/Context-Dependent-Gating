### Authors: Nicolas Y. Masse, Gregory D. Grant


import numpy as np
from parameters import *
import pickle
import matplotlib.pyplot as plt


class Stimulus:

    def __init__(self, include_cifar10 = False, cifar_labels_per_task = 5):
        if par['task'] == 'mnist':
            self.generate_mnist_tuning()
        elif par['task'] == 'cifar':
            self.cifar10_dir = './cifar/cifar-10-python/'
            self.cifar100_dir = './cifar/cifar-100-python/'
            self.num_cifar_labels = 110
            self.cifar_labels_per_task = cifar_labels_per_task
            self.generate_cifar_tuning(include_cifar10)
            self.find_cifar_indices()


    def generate_mnist_tuning(self):
        from mnist import MNIST
        mndata = MNIST('./mnist/data/original')
        self.mnist_train_images, self.mnist_train_labels = mndata.load_training()
        self.mnist_test_images,  self.mnist_test_labels  = mndata.load_testing()

        self.num_train_examples = len(self.mnist_train_images)
        self.num_test_examples  = len(self.mnist_test_images)
        self.num_outputs        = 10

        self.mnist_train_images = np.array(self.mnist_train_images)/255
        self.mnist_test_images  = np.array(self.mnist_test_images)/255

        self.mnist_permutation  = []
        for t in range(par['n_tasks']):
            self.mnist_permutation.append(np.random.permutation(784))


    def generate_cifar_tuning(self, include_cifar10 = False):

        self.cifar_train_images = np.array([])
        self.cifar_train_labels = np.array([])
        self.cifar_test_images  = np.array([])
        self.cifar_test_labels  = np.array([])

        """
        Load CIFAR-10 data
        """
        if include_cifar10:
            ########################
            ### Currently unused ###
            for i in range(5):
                x =  pickle.load(open(self.cifar10_dir + 'data_batch_' + str(i+1),'rb'), encoding='bytes')
                self.cifar_train_images = np.vstack((self.cifar_train_images, x[b'data'])) if self.cifar_train_images.size else x[b'data']
                labels = np.reshape(np.array(x[b'labels']),(-1,1))
                self.cifar_train_labels = np.vstack((self.cifar_train_labels, labels))  if self.cifar_train_labels.size else labels

            x =  pickle.load(open(self.cifar10_dir + 'test_batch','rb'), encoding='bytes')
            self.cifar_test_images = np.array(x[b'data'])
            self.cifar_test_labels = np.reshape(np.array(x[b'labels']),(-1,1))
            ### Currently unused ###
            ########################

        """
        Load CIFAR-100 data
        """
        x = pickle.load(open(self.cifar100_dir + 'train','rb'), encoding='bytes')
        labels = np.reshape(np.array(x[b'fine_labels']),(-1,1))
        images = x[b'data']

        self.cifar_train_images = np.vstack((self.cifar_train_images, images)) if self.cifar_train_images.size else np.array(images)
        self.cifar_train_labels = np.vstack((self.cifar_train_labels, labels)) if self.cifar_train_labels.size else np.array(labels)

        x = pickle.load(open(self.cifar100_dir + 'test','rb'), encoding='bytes')
        labels = np.reshape(np.array(x[b'fine_labels']),(-1,1))
        images = x[b'data']

        self.cifar_test_images  = np.vstack((self.cifar_test_images, images))  if self.cifar_test_images.size else np.array(images)
        self.cifar_test_labels  = np.vstack((self.cifar_test_labels, labels))  if self.cifar_test_labels.size else np.array(labels)

        print('CIFAR shapes:', self.cifar_test_labels.shape, self.cifar_train_labels.shape)


    def find_cifar_indices(self):

        self.cifar_train_ind = []
        self.cifar_test_ind  = []
        for i in range(0, self.num_cifar_labels, self.cifar_labels_per_task):
            self.cifar_train_ind.append(np.where((self.cifar_train_labels>=i)*(self.cifar_train_labels<i+self.cifar_labels_per_task))[0])
            self.cifar_test_ind.append(np.where((self.cifar_test_labels>=i)*(self.cifar_test_labels<i+self.cifar_labels_per_task))[0])


    def generate_cifar_batch(self, task_num, test = False):

        # Select example indices
        ind_ref = self.cifar_test_ind if test else self.cifar_train_ind
        ind = ind_ref[task_num]
        q = np.random.randint(0,len(ind),par['batch_size'])

        # Pick out batch data and labels
        batch_data   = np.zeros((par['batch_size'], 32,32,3), dtype = np.float32)
        batch_labels = np.zeros((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)
        if par['multihead']:
            mask = np.zeros((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)
            mask[:, task_num*self.cifar_labels_per_task:(task_num+1)*self.cifar_labels_per_task] = 1
        else:
            mask = np.ones((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)

        for i in range(par['batch_size']):
            if test:
                if par['multihead']:
                    k = int(self.cifar_test_labels[ind[q[i]]])
                else:
                    k = self.cifar_test_labels[ind[q[i]]][0]%self.cifar_labels_per_task

                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.cifar_test_images[ind[q[i]], :],(1,32,32,3), order='F'))/255
            else:
                if par['multihead']:
                    k = int(self.cifar_train_labels[ind[q[i]]])
                else:
                    k = self.cifar_train_labels[ind[q[i]]][0]%self.cifar_labels_per_task

                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.cifar_train_images[ind[q[i]], :],(1,32,32,3), order='F'))/255

        return batch_data, batch_labels, mask


    def generate_mnist_batch(self, task_num, test = False):

        # Select random example indices
        ind_num = self.num_test_examples if test else self.num_train_examples
        q = np.random.randint(0, ind_num, par['batch_size'])

        # Pick out batch data and labels
        batch_data   = np.zeros((par['batch_size'], 28**2), dtype=np.float32)
        batch_labels = np.zeros((par['batch_size'], self.num_outputs), dtype=np.float32)
        for i in range(par['batch_size']):
            if test:
                k = self.mnist_test_labels[q[i]]
                batch_labels[i, k] = 1
                batch_data[i, :] = self.mnist_test_images[q[i]][self.mnist_permutation[task_num]]
            else:
                k = self.mnist_train_labels[q[i]]
                batch_labels[i, k] = 1
                batch_data[i, :] = self.mnist_train_images[q[i]][self.mnist_permutation[task_num]]

        return batch_data, batch_labels


    def make_batch(self, task_num, test=False):

        # Allow for random interleaving
        task_num = np.random.randint(par['n_tasks']) if task_num<0 else task_num

        if par['task'] == 'mnist':
            batch_data, batch_labels = self.generate_mnist_batch(task_num, test)
            mask = np.ones((par['batch_size'], 10), dtype=np.float32)
        elif par['task'] == 'cifar':
            batch_data, batch_labels, mask = self.generate_cifar_batch(task_num, test)
        else:
            raise Exception('Unrecognized task')


        return batch_data, batch_labels, mask
