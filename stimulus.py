### Authors: Nicolas Y. Masse, Gregory D. Grant


import numpy as np
from parameters import *
import pickle
import matplotlib.pyplot as plt


class Stimulus:

    def __init__(self, include_cifar10 = False, labels_per_task = 5, include_all = False):
        if par['task'] == 'mnist':
            self.generate_mnist_tuning()
        elif par['task'] == 'cifar':
            self.cifar10_dir = './cifar/cifar-10-python/'
            self.cifar100_dir = './cifar/cifar-100-python/'
            self.num_labels = 110
            self.labels_per_task = labels_per_task
            self.generate_cifar_tuning(include_cifar10, include_all)
            self.find_indices()

        elif par['task'] == 'imagenet':
            self.imagenet_dir = './ImageNet/'
            self.num_labels = 1000
            self.labels_per_task = labels_per_task
            self.generate_imagenet_tuning()
            self.find_indices()


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

    def generate_imagenet_tuning(self):

        self.train_images = np.array([])
        self.train_labels = np.array([])

        for i in range(10):
            x =  pickle.load(open(self.imagenet_dir + 'train_data_batch_' + str(i+1),'rb'))
            self.train_images = np.vstack((self.train_images, x['data'])) if self.train_images.size else x['data']
            labels = np.reshape(np.array(x['labels']),(-1,1))
            self.train_labels = np.vstack((self.train_labels, labels))  if self.train_labels.size else labels

        x =  pickle.load(open(self.imagenet_dir + 'val_data','rb'))
        self.test_images = np.array(x['data'])
        self.test_labels = np.reshape(np.array(x['labels']),(-1,1))


    def generate_cifar_tuning(self, include_cifar10 = False, include_all = False):

        self.train_images = np.array([])
        self.train_labels = np.array([])
        self.test_images = np.array([])
        self.test_labels = np.array([])

        """
        Load CIFAR-10 data
        """
        if include_cifar10:
            for i in range(5):
                x =  pickle.load(open(self.cifar10_dir + 'data_batch_' + str(i+1),'rb'), encoding='bytes')
                self.train_images = np.vstack((self.train_images, x[b'data'])) if self.train_images.size else x[b'data']
                labels = np.reshape(np.array(x[b'labels']),(-1,1))
                self.train_labels = np.vstack((self.train_labels, labels))  if self.train_labels.size else labels

            x =  pickle.load(open(self.cifar10_dir + 'test_batch','rb'), encoding='bytes')
            self.test_images = np.array(x[b'data'])
            self.test_labels = np.reshape(np.array(x[b'labels']),(-1,1))
            if include_all:
                # used for training the convolutional layers
                #use both training and testing data
                self.train_images = np.vstack((self.train_images, x[b'data']))
                self.train_labels = np.vstack((self.train_labels,  np.reshape(np.array(x[b'labels']),(-1,1))))


        """
        Load CIFAR-100 data
        """
        x = pickle.load(open(self.cifar100_dir + 'train','rb'), encoding='bytes')
        labels = np.reshape(np.array(x[b'fine_labels']),(-1,1))

        self.train_images = np.vstack((self.train_images, x[b'data'])) if self.train_images.size else np.array(images)
        self.train_labels = np.vstack((self.train_labels, labels)) if self.train_labels.size else np.array(labels)

        x = pickle.load(open(self.cifar100_dir + 'test','rb'), encoding='bytes')
        labels = np.reshape(np.array(x[b'fine_labels']),(-1,1))

        self.test_images  = np.vstack((self.test_images, x[b'data']))  if self.test_images.size else np.array(images)
        self.test_labels  = np.vstack((self.test_labels, labels))  if self.test_labels.size else np.array(labels)

        if include_all:
            # used for training the convolutional layers
            #use both training and testing data
            self.train_images = np.vstack((self.train_images, x[b'data']))
            self.train_labels = np.vstack((self.train_labels, labels))

        print('CIFAR shapes:', self.test_labels.shape, self.train_labels.shape)


    def find_indices(self):

        self.train_ind = []
        self.test_ind  = []
        for i in range(0, self.num_labels, self.labels_per_task):
            self.train_ind.append(np.where((self.train_labels>=i)*(self.train_labels<i+self.labels_per_task))[0])
            self.test_ind.append(np.where((self.test_labels>=i)*(self.test_labels<i+self.labels_per_task))[0])


    def generate_image_batch(self, task_num, test = False):

        # Select example indices
        ind_ref = self.test_ind if test else self.train_ind
        ind = ind_ref[task_num]
        q = np.random.randint(0,len(ind),par['batch_size'])

        # Pick out batch data and labels
        batch_data   = np.zeros((par['batch_size'], 32,32,3), dtype = np.float32)
        batch_labels = np.zeros((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)
        if par['multihead']:
            mask = np.zeros((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)
            mask[:, task_num*self.labels_per_task:(task_num+1)*self.labels_per_task] = 1
        else:
            mask = np.ones((par['batch_size'], par['layer_dims'][-1]), dtype = np.float32)

        for i in range(par['batch_size']):
            if test:
                if par['multihead']:
                    k = int(self.test_labels[ind[q[i]]])
                else:
                    k = self.test_labels[ind[q[i]]][0]%self.labels_per_task
                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.test_images[ind[q[i]], :],(1,32,32,3), order='F'))/255
            else:
                if par['multihead']:
                    k = int(self.train_labels[ind[q[i]]])
                else:
                    k = self.train_labels[ind[q[i]]][0]%self.labels_per_task
                batch_labels[i, k] = 1
                batch_data[i, :] = np.float32(np.reshape(self.train_images[ind[q[i]], :],(1,32,32,3), order='F'))/255

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
        elif par['task'] == 'cifar' or par['task'] == 'imagenet':
            batch_data, batch_labels, mask = self.generate_image_batch(task_num, test)
        else:
            raise Exception('Unrecognized task')


        return batch_data, batch_labels, mask
