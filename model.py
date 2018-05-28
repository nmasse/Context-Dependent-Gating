### Authors: Nicolas Y. Masse, Gregory D. Grant

import tensorflow as tf
import numpy as np
import stimulus
import AdamOpt
from parameters import *
import os, time
import pickle
import convolutional_layers
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################
class Model:

    def __init__(self, input_data, target_data, gating, mask, droput_keep_pct, input_droput_keep_pct):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data         = input_data
        self.gating             = gating
        self.target_data        = target_data
        self.droput_keep_pct    = droput_keep_pct
        self.input_droput_keep_pct  = input_droput_keep_pct
        self.mask               = mask

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        if par['task'] == 'cifar' or par['task'] == 'imagenet':
            self.x = self.apply_convulational_layers()

        elif par['task'] == 'mnist':
            self.x = tf.nn.dropout(self.input_data, self.input_droput_keep_pct)

        self.apply_dense_layers()


    def apply_dense_layers(self):

        for n in range(par['n_layers']-1):
            with tf.variable_scope('layer'+str(n)):

                W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], \
                    -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                b = tf.get_variable('b', initializer = tf.zeros([1,par['layer_dims'][n+1]]), trainable = True if n<par['n_layers']-2 else False)

                if n < par['n_layers']-2:
                    self.x = tf.nn.dropout(tf.nn.relu(tf.matmul(self.x,W) + b), self.droput_keep_pct)
                    self.x = self.x*tf.tile(tf.reshape(self.gating[n],[1,par['layer_dims'][n+1]]),[par['batch_size'],1])

                else:
                    self.y = tf.matmul(self.x,W) + b  - (1-self.mask)*1e16


    def apply_convulational_layers(self):

        conv_weights = pickle.load(open(par['save_dir'] + par['task'] + '_conv_weights.pkl','rb'))
        #conv_weights = pickle.load(open(par['save_dir'] + 'cifarconv_weights.pkl','rb'))

        conv1 = tf.layers.conv2d(inputs=self.input_data,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv2 = tf.layers.conv2d(inputs=conv1,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_1/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_1/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.input_droput_keep_pct)

        conv3 = tf.layers.conv2d(inputs=conv2,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_2/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_2/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv4 = tf.layers.conv2d(inputs=conv3,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_3/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_3/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='SAME')
        conv4 = tf.nn.dropout(conv4, self.input_droput_keep_pct)

        return tf.reshape(conv4,[par['batch_size'], -1])


    def optimize(self):

        # Use all trainable variables, except those in the convolutional layers
        self.variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = par['learning_rate'])

        previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        for var in self.variables:
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            aux_losses.append(par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
               tf.square(previous_weights_mu_minus_1[var.op.name] - var) )))
            reset_prev_vars_ops.append( tf.assign(previous_weights_mu_minus_1[var.op.name], var ) )

        self.aux_loss = tf.add_n(aux_losses)

        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, \
            labels = self.target_data, dim=1))

        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.task_loss, self.aux_loss]):
            self.train_op = adam_optimizer.compute_gradients(self.task_loss + self.aux_loss)

        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(adam_optimizer, previous_weights_mu_minus_1)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC()

        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()

        correct_prediction = tf.equal(tf.argmax(self.y - (1-self.mask)*9999,1), tf.argmax(self.target_data - (1-self.mask)*9999,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.reset_weights()

    def reset_weights(self):

        reset_weights = []

        for var in self.variables:
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights.append(tf.assign(var, var*0.))
            elif 'W' in var.op.name:
                # reset weights to uniform randomly distributed
                layer = int(var.op.name[5])
                new_weight = tf.random_uniform([par['layer_dims'][layer],par['layer_dims'][layer+1]], \
                    -1.0/np.sqrt(par['layer_dims'][layer]), 1.0/np.sqrt(par['layer_dims'][layer]))
                reset_weights.append(tf.assign(var,new_weight))

        self.reset_weights = tf.group(*reset_weights)

    def EWC(self):

        # Kirkpatrick method
        epsilon = 1e-5
        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(1.)

        # sample label from logits
        class_ind = tf.multinomial(self.y, 1)
        # model results p(y|x, theta)
        p_theta = tf.nn.softmax(self.y, dim = 1)

        class_ind_one_hot = tf.reshape(tf.one_hot(class_ind, par['layer_dims'][-1]), \
            [par['batch_size'], par['layer_dims'][-1]])
        # calculate the gradient of log p(y|x, theta)
        log_p_theta = tf.unstack(class_ind_one_hot*tf.log(p_theta + epsilon), axis = 0)
        for lp in log_p_theta:
            grads_and_vars = opt.compute_gradients(lp)
            for grad, var in grads_and_vars:
                fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                    grad*grad/par['batch_size']/par['EWC_fisher_num_batches']))

        self.update_big_omega = tf.group(*fisher_ops)

    def pathint_stabilization(self, adam_optimizer, previous_weights_mu_minus_1):
        # Zenke method

        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in self.variables:

            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))


        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # This is called every batch
        with tf.control_dependencies([self.train_op]):
            self.delta_grads = adam_optimizer.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.task_loss)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

def main(save_fn, gpu_id = None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # train the convolutional layers with the CIFAR-10 dataset
    # otherwise, it will load the convolutional weights from the saved file
    if (par['task'] == 'cifar' or par['task'] == 'imagenet') and par['train_convolutional_layers']:
        convolutional_layers.ConvolutionalLayers()

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, target_data, gating, mask, dropout keep pct hidden layers, dropout keep pct input layers

    if par['task'] == 'mnist':
        x  = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][0]], 'stim')
    elif par['task'] == 'cifar' or par['task'] == 'imagenet':
        x  = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'stim')
    y   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'out')
    mask   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'mask')
    droput_keep_pct = tf.placeholder(tf.float32, [], 'dropout')
    input_droput_keep_pct = tf.placeholder(tf.float32, [], 'input_dropout')
    gating = [tf.placeholder(tf.float32, [par['layer_dims'][n+1]], 'gating') for n in range(par['n_layers']-1)]

    stim = stimulus.Stimulus(labels_per_task = par['labels_per_task'])
    accuracy_full = []
    accuracy_grid = np.zeros((par['n_tasks'], par['n_tasks']))

    with tf.Session() as sess:

        if gpu_id is None:
            model = Model(x, y, gating, mask, droput_keep_pct, input_droput_keep_pct)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, y, gating, mask, droput_keep_pct, input_droput_keep_pct)
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        for task in range(par['n_tasks']):

            # create dictionary of gating signals applied to each hidden layer for this task
            gating_dict = {k:v for k,v in zip(gating, par['gating'][task])}

            for i in range(par['n_train_batches']):

                # make batch of training data
                stim_in, y_hat, mk = stim.make_batch(task, test = False)

                if par['stabilization'] == 'pathint':

                    _, _, loss, AL = sess.run([model.train_op, model.update_small_omega, model.task_loss, model.aux_loss], \
                        feed_dict = {x:stim_in, y:y_hat, **gating_dict, mask:mk, droput_keep_pct:par['drop_keep_pct'], \
                        input_droput_keep_pct:par['input_drop_keep_pct']})

                elif par['stabilization'] == 'EWC':
                    _,loss,AL = sess.run([model.train_op, model.task_loss, model.aux_loss], feed_dict = \
                        {x:stim_in, y:y_hat, **gating_dict, mask:mk, droput_keep_pct:par['drop_keep_pct'], input_droput_keep_pct:par['input_drop_keep_pct']})

                if i//500 == i/500:
                    print('Iter: ', i, 'Loss: ', loss, 'Aux Loss: ',  AL)

            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                big_omegas = sess.run([model.update_big_omega, model.big_omega_var])
            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    stim_in, y_hat, mk = stim.make_batch(task, test = False)
                    big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = \
                        {x:stim_in, y:y_hat, **gating_dict, mask:mk, droput_keep_pct:1.0, input_droput_keep_pct:1.0})

            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            # Test the netwroks on all trained tasks
            num_test_reps = 10
            accuracy = np.zeros((task+1))
            for test_task in range(task+1):
                gating_dict = {k:v for k,v in zip(gating, par['gating'][test_task])}
                for r in range(num_test_reps):
                    stim_in, y_hat, mk = stim.make_batch(test_task, test = True)
                    acc = sess.run(model.accuracy, feed_dict={x:stim_in, y:y_hat, \
                        **gating_dict, mask:mk, droput_keep_pct:1.0, input_droput_keep_pct:1.0})/num_test_reps
                    accuracy_grid[task, test_task]  += acc
                    accuracy[test_task] += acc

            print('Task ',task, ' Mean ', np.mean(accuracy), ' First ', accuracy[0], ' Last ', accuracy[-1])
            accuracy_full.append(np.mean(accuracy))

            # reset weights between tasks if called upon
            if par['reset_weights']:
                sess.run(model.reset_weights)


        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                            'accuracy_grid': accuracy_grid, 'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nModel execution complete.')
