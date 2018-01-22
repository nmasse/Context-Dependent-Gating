### Authors: Nicolas Y. Masse, Gregory D. Grant

import tensorflow as tf
import numpy as np
import stimulus
import AdamOpt
from parameters import *
import os, time
import pickle
import top_down
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi


# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

###################
### Model setup ###
###################
class Model:

    def __init__(self, input_data, td_data, target_data, mask, droput_keep_pct, input_droput_keep_pct):

        # Load the input activity, the target data, and the training mask
        # for this batch of trials
        self.input_data         = input_data
        self.td_data            = td_data
        self.target_data        = target_data
        self.droput_keep_pct    = droput_keep_pct
        self.input_droput_keep_pct  = input_droput_keep_pct
        self.mask               = mask

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):

        if par['task'] == 'cifar':
            self.x = self.apply_convulational_layers()

        elif par['task'] == 'mnist':
            self.x = tf.nn.dropout(self.input_data, self.input_droput_keep_pct)
            #self.x = tf.nn.dropout(self.input_data, 1.0)

        self.apply_dense_layers()


    def apply_dense_layers(self):

        self.neuron_activity = []
        self.dendrite_activity = []

        self.spike_loss = 0
        for n, scope_name in enumerate(['layer'+str(n) for n in range(par['n_layers']-1)]):
            with tf.variable_scope(scope_name):
                if n < par['n_layers']-2 or par['dendrites_final_layer']:
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['n_dendrites'],par['layer_dims'][n+1]], \
                        -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['n_dendrites'],par['layer_dims'][n+1]]), trainable = True)
                    W_td = tf.get_variable('W_td', initializer = par['W_td0'][n], trainable = False)

                    proj_W_td = tf.tensordot(self.td_data, W_td, ([1],[0]))
                    proj_W_td = tf.tile(proj_W_td,[par['batch_size'], 1, 1])
                    #W_effective = W*tf.tile(proj_W_td,[par['layer_dims'][n],1,1])
                    #b_effective = b*proj_W_td

                else:
                    # final layer -> no dendrites
                    W = tf.get_variable('W', initializer = tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], \
                        -1/np.sqrt(par['layer_dims'][n]), 1/np.sqrt(par['layer_dims'][n])), trainable = True)
                    b = tf.get_variable('b', initializer = tf.zeros([1,par['layer_dims'][n+1]]), trainable = True)


                if n < par['n_layers']-2:
                    #dend_activity = tf.nn.relu(tf.tensordot(self.x, W_effective, ([1],[0]))  + b_effective)
                    dend_activity = tf.nn.relu(tf.tensordot(self.x, W, ([1],[0]))  + b)
                    dend_activity = proj_W_td*dend_activity
                    print('proj_W_td',proj_W_td)
                    print('dend_activity', dend_activity)
                    self.x = tf.nn.dropout(tf.reduce_sum(dend_activity, axis=1), self.droput_keep_pct)
                    self.spike_loss += tf.reduce_sum(self.x)
                    self.neuron_activity.append(self.x)
                    self.dendrite_activity.append(dend_activity)

                else:
                    if par['dendrites_final_layer']:
                        dend_activity = tf.tensordot(self.x, W, ([1],[0])) + b
                        dend_activity = proj_W_td*dend_activity
                        # Want to ensure that masked out values don't contribute to softmax
                        self.y = self.mask*tf.nn.softmax(tf.reduce_sum(dend_activity, axis=1) - (1-self.mask)*1e32, dim = 1)
                        self.dendrite_activity.append(dend_activity)

                    else:
                        self.y = tf.nn.softmax(tf.matmul(self.x,W) + b  - (1-self.mask)*1e32 , dim = 1)
                    self.neuron_activity.append(self.y)

    def apply_convulational_layers(self):

        conv_weights = pickle.load(open('./encoder_testing/conv_weights.pkl','rb'))

        conv1 = tf.layers.conv2d(inputs=self.input_data,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv2 = tf.layers.conv2d(inputs=conv1,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_1/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_1/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')

        conv3 = tf.layers.conv2d(inputs=conv2,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_2/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_2/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv4 = tf.layers.conv2d(inputs=conv3,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_3/kernel']),  bias_initializer = tf.constant_initializer(conv_weights['conv2d_3/bias']), \
            strides=1, activation=tf.nn.relu, padding = 'SAME', trainable=False)

        conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='SAME')

        return tf.reshape(conv4,[par['batch_size'], -1])


    def optimize(self):

        epsilon = 1e-4

        # Use all trainable variables, except those in the convolutional layers
        #variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        adam_optimizer = AdamOpt.AdamOpt(variables, learning_rate = par['learning_rate'])

        previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        self.aux_loss = 0.0

        for var in variables:
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.aux_loss += par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
                tf.square(previous_weights_mu_minus_1[var.op.name] - var) ))
            reset_prev_vars_ops.append( tf.assign(previous_weights_mu_minus_1[var.op.name], var ) )

        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, \
            labels = self.target_data, dim=1))


        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        self.spike_loss /= np.sum(par['layer_dims'][1:-1]) # currently not in use
        self.train_op = adam_optimizer.compute_gradients(self.task_loss + self.aux_loss)

        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(variables, adam_optimizer, previous_weights_mu_minus_1)

        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC(variables)


        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()

        correct_prediction = tf.equal(tf.argmax(self.y - (1-self.mask)*9999,1), tf.argmax(self.target_data - (1-self.mask)*9999,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def EWC(self, variables):
        # Kirkpatrick method
        epsilon = 1e-9

        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(1)
        y_unstacked = tf.unstack(self.y, axis = 0)
        for y1 in tf.unstack(y_unstacked[:par['EWC_fisher_calc_batch']]):
            grads_and_vars = opt.compute_gradients(tf.log(y1 + epsilon))
            for grad, var in grads_and_vars:
                print(var.op.name, grad)
                fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], \
                    grad*grad/par['EWC_fisher_calc_batch']/par['EWC_fisher_num_batches']/par['layer_dims'][-1]))

        self.update_big_omega = tf.group(*fisher_ops)

    def pathint_stabilization(self, variables, adam_optimizer, previous_weights_mu_minus_1):
        # Zenke method

        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  par['learning_rate'])
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in variables:

            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))


        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        #with tf.control_dependencies([self.train_op]):
        self.delta_grads = adam_optimizer.return_delta_grads()
        self.gradients = optimizer_task.compute_gradients(self.task_loss)
        # This is called every batch
        for grad,var in self.gradients:
            update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )

        self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!

def main(save_fn, gpu_id):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    #determine_top_down_weights()
    if par['task'] == 'cifar' and par['train_convolutional_layers']:
        top_down.ConvolutionalLayers()

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    # input_data, td_data, target_data, learning_rate, stim_train
    if par['task'] == 'mnist':
        x  = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][0]], 'stim')
    elif par['task'] == 'cifar':
        x  = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'stim')
    td  = tf.placeholder(tf.float32, [1, par['n_td']], 'TD')
    y   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'out')
    mask   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'mask')
    droput_keep_pct = tf.placeholder(tf.float32, [], 'dropout')
    input_droput_keep_pct = tf.placeholder(tf.float32, [], 'input_dropout')


    stim = stimulus.Stimulus()
    accuracy_full = []

    with tf.Session() as sess:

        with tf.device("/gpu:0"):
            model = Model(x, td, y, mask, droput_keep_pct, input_droput_keep_pct)
            init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()
        sess.run(model.reset_prev_vars)

        for task in range(par['n_tasks']):

            for i in range(par['n_train_batches']):

                stim_in, y_hat, td_in, mk = stim.make_batch(task, test = False)
                if par['stabilization'] == 'pathint':

                    # Breaking session.run into two steps appears is important
                    _,loss,spike_loss,AL = sess.run([model.train_op, model.task_loss,model.spike_loss, model.aux_loss], feed_dict = \
                        {x:stim_in, td:td_in, y:y_hat, mask:mk, droput_keep_pct:par['drop_keep_pct'], input_droput_keep_pct:par['input_drop_keep_pct']})
                    _,dg = sess.run([model.update_small_omega, model.delta_grads], feed_dict = \
                        {x:stim_in, td:td_in, y:y_hat, mask:mk, droput_keep_pct:par['drop_keep_pct'], input_droput_keep_pct:par['input_drop_keep_pct']})


                elif par['stabilization'] == 'EWC':
                    _,loss,spike_loss,AL = sess.run([model.train_op, model.task_loss,model.spike_loss, model.aux_loss], feed_dict = \
                        {x:stim_in, td:td_in, y:y_hat, mask:mk, droput_keep_pct:par['drop_keep_pct'], input_droput_keep_pct:par['input_drop_keep_pct']})

                if i//1000 == i/1000:
                    # Print results to screen
                    print('Iter: ', i, 'Loss: ', loss, 'Aux Loss: ',  AL, 'Spike cost:', spike_loss)

            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                #sess.run(model.update_big_omega,feed_dict={x:stim_in, td:td_in, mask:mk, droput_keep_pct:1.0})
                big_omegas = sess.run([model.update_big_omega, model.big_omega_var])
            elif par['stabilization'] == 'EWC':
                for n in range(par['EWC_fisher_num_batches']):
                    stim_in, y_hat, td_in, mk = stim.make_batch(task, test = False)
                    big_omegas = sess.run([model.update_big_omega,model.big_omega_var], feed_dict = \
                        {x:stim_in, td:td_in, mask:mk, droput_keep_pct:1.0, input_droput_keep_pct:1.0})

            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            num_test_reps = 10
            accuracy = np.zeros((task+1))
            for test_task in range(task+1):
                for r in range(num_test_reps):
                    stim_in, y_hat, td_in, mk = stim.make_batch(test_task, test = True)
                    accuracy[test_task] += sess.run(model.accuracy, feed_dict={x:stim_in, \
                        td:td_in, y:y_hat, mask:mk,droput_keep_pct:1.0, input_droput_keep_pct:1.0})/num_test_reps

            print('Task ',task, ' Mean ', np.mean(accuracy), ' First ', accuracy[0], ' Last ', accuracy[-1])
            accuracy_full.append(np.mean(accuracy))

        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                'big_omegas': big_omegas, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nModel execution complete.')
