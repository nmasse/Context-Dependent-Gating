### Context-Dependent Gating Feed-Forward Neural Network, for paper:
### Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization
### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys

# Model modules
from parameters import *
import stimulus
import AdamOpt
import convolutional_layers

# Make GPU IDs match nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore startup TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:

    """ Feedforward model with optional convolutional network
        for supervised learning training """

    def __init__(self, input_data, target_data, gating, mask, droput_keep_pct, input_droput_keep_pct, rule):

        # Load the input activity, the target data, training mask, etc.
        self.input_data             = input_data
        self.gating                 = gating
        self.target_data            = target_data
        self.droput_keep_pct        = droput_keep_pct
        self.input_droput_keep_pct  = input_droput_keep_pct
        self.mask                   = mask
        self.rule                   = rule
        self.conv_droput_keep_pct   = (1+self.droput_keep_pct)/2

        # Build the Tensorglow graph
        self.run_model()

        # Train the model
        self.optimize()


    def run_model(self):
        """ Depending on the task, generate the input to the feedforward model
            either through a convolutional network or with dropout. """

        # Apply input condition
        if par['task'] == 'cifar' or par['task'] == 'imagenet':
            self.x = self.apply_convolutional_layers()
        elif par['task'] == 'mnist':
            self.x = tf.nn.dropout(self.input_data, self.input_droput_keep_pct)

        # Apply feedforward network
        self.apply_dense_layers()


    def apply_dense_layers(self):
        """ Run the feedforward part of the model by iterating through
            the hidden layers """

        # Iterate over hidden layers
        for n in range(par['n_layers']-1):

            # Select layer scope
            with tf.variable_scope('layer'+str(n)):

                # Generate weight and bias for this layer
                W = tf.get_variable('W', initializer=tf.random_uniform([par['layer_dims'][n],par['layer_dims'][n+1]], \
                    -1.0/np.sqrt(par['layer_dims'][n]), 1.0/np.sqrt(par['layer_dims'][n])), trainable=True)
                b = tf.get_variable('b', initializer=tf.zeros([1,par['layer_dims'][n+1]]), trainable = True)

                # If this is not the last hidden layer, proceed normally.  If
                # it is, generate the output
                if n < par['n_layers']-2:

                    # If including the rule signal, generate a weight matrix
                    # from the rule to the current hidden layer
                    if par['include_rule_signal']:
                        Wr = tf.get_variable('Wr', initializer=tf.random_uniform([par['n_tasks'], par['layer_dims'][n+1]], \
                            -1.0/np.sqrt(par['n_tasks']), 1.0/np.sqrt(par['n_tasks'])), trainable=True)
                        r = tf.matmul(self.rule, Wr)
                    else:
                        r = tf.constant(0.)

                    # Make the input to the next hidden layer, accounting for
                    # dropout, rule terms, and gating vectors
                    self.x = tf.nn.dropout(tf.nn.relu(tf.matmul(self.x,W) + r + b), self.droput_keep_pct)
                    self.x = self.x*tf.tile(tf.reshape(self.gating[n],[1,par['layer_dims'][n+1]]),[par['batch_size'],1])

                else:

                    # Make output vector
                    self.y = tf.matmul(self.x,W) + b

                    # Apply strong negative values to non-active outputs
                    # in the case of the multihead network configuration
                    if par['multihead']:
                        self.y -= (1-self.mask)*1e16


    def apply_convolutional_layers(self):
        """ Run the convolutional part of the model to reduce 32 x 32 x 3
            inputs to a smaller and more interpretable input vector.  The
            convolutional weights are loaded from a file in par['save_dir']. """

        # Load weights
        conv_weights = pickle.load(open(par['save_dir'] + 'conv_weights.pkl','rb'))

        # Apply first two convolutional layers
        conv1 = tf.layers.conv2d(inputs=self.input_data,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d/kernel']),  bias_initializer = \
            tf.constant_initializer(conv_weights['conv2d/bias']),  strides=1, activation=tf.nn.relu, \
            padding = 'SAME', trainable=False)

        conv2 = tf.layers.conv2d(inputs=conv1,filters=32, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_1/kernel']),  bias_initializer = \
            tf.constant_initializer(conv_weights['conv2d_1/bias']),  strides=1, activation=tf.nn.relu, \
            padding = 'SAME', trainable=False)

        # Apply max pooling and dropout
        conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.conv_droput_keep_pct)

        # Apply next two convolutional layers
        conv3 = tf.layers.conv2d(inputs=conv2,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_2/kernel']),  bias_initializer = \
            tf.constant_initializer(conv_weights['conv2d_2/bias']), strides=1, activation=tf.nn.relu, \
            padding = 'SAME', trainable=False)

        conv4 = tf.layers.conv2d(inputs=conv3,filters=64, kernel_size=[3, 3], kernel_initializer = \
            tf.constant_initializer(conv_weights['conv2d_3/kernel']),  bias_initializer = \
            tf.constant_initializer(conv_weights['conv2d_3/bias']), strides=1, activation=tf.nn.relu, \
            padding = 'SAME', trainable=False)

        # Apply max pooling and dropout
        conv4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, padding='SAME')
        conv4 = tf.nn.dropout(conv4, self.conv_droput_keep_pct)

        # Return flattened inputs
        return tf.reshape(conv4,[par['batch_size'], -1])


    def optimize(self):
        """ Calculate losses and apply corrections to the model """

        # Optimize all trainable variables, except those in the convolutional layers
        self.variables = [var for var in tf.trainable_variables() if not 'conv' in var.op.name]

        # Use all trainable variables for synaptic stabilization, except conv and rule weights
        self.variables_stabilization = [var for var in tf.trainable_variables() if not ('conv' in var.op.name or 'Wr' in var.op.name)]

        # Set up the optimizer
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = par['learning_rate'])

        # Make stabilization records
        prev_weights = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        # Set up stabilization based on designated variables list
        for var in self.variables_stabilization:
            n = var.op.name

            # Make big omega and prev_weight variables
            self.big_omega_var[n] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            prev_weights[n] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

            # Generate auxiliary stabilization losses
            aux_losses.append(par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[n], tf.square(prev_weights[n] - var))))

            # Make a reset function for each prev_weight element
            reset_prev_vars_ops.append(tf.assign(prev_weights[n], var))

        # Aggregate auxiliary losses
        self.aux_loss = tf.add_n(aux_losses)

        # Determine softmax task loss on the network output
        self.task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, \
            labels = self.target_data, dim=1))

        # Get the gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.task_loss, self.aux_loss]):
            self.train_op = adam_optimizer.compute_gradients(self.task_loss + self.aux_loss)

        # Stabilize weights
        if par['stabilization'] == 'pathint':
            # Zenke method
            self.pathint_stabilization(adam_optimizer, prev_weights)
        elif par['stabilization'] == 'EWC':
            # Kirkpatrick method
            self.EWC()
        else:
            # No stabilization
            pass

        # Make reset operations
        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()
        self.reset_weights()

        # Calculate accuracy for analysis
        correct_prediction = tf.equal(tf.argmax(self.y - (1-self.mask)*9999,1), tf.argmax(self.target_data - (1-self.mask)*9999,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def reset_weights(self):
        """ Make new weights, if requested """

        reset_weights = []

        for var in self.variables:
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights.append(tf.assign(var, var*0.))
            elif 'Wr' in var.op.name:
                # reset rule weights to uniform randomly distributed
                # (albeit with different shapes than standard weights)
                layer = int(var.op.name[5])
                new_weight = tf.random_uniform([par['n_tasks'],par['layer_dims'][layer+1]], \
                    -1.0/np.sqrt(par['n_tasks']), 1.0/np.sqrt(par['n_tasks']))
                reset_weights.append(tf.assign(var,new_weight))
            elif 'W' in var.op.name:
                # reset weights to uniform randomly distributed
                layer = int(var.op.name[5])
                new_weight = tf.random_uniform([par['layer_dims'][layer],par['layer_dims'][layer+1]], \
                    -1.0/np.sqrt(par['layer_dims'][layer]), 1.0/np.sqrt(par['layer_dims'][layer]))
                reset_weights.append(tf.assign(var,new_weight))

        self.reset_weights = tf.group(*reset_weights)

    def EWC(self):
        """ Synaptic stabilization via the Kirkpatrick method """

        # Set up method
        epsilon = 1e-6
        fisher_ops = []
        opt = tf.train.GradientDescentOptimizer(1.)

        # Sample label from logits
        p_theta = tf.nn.softmax(self.y, dim = 1)
        class_ind_one_hot = tf.cast(tf.squeeze(tf.one_hot(tf.multinomial(self.y, 1), par['layer_dims'][-1])), tf.float32)
        log_p_theta = tf.unstack(class_ind_one_hot*tf.log(p_theta + epsilon), axis = 0)

        # Iterate over the variables to get gradient shapes
        grad_dict = {}
        for var in self.variables_stabilization:
            grad_dict[var.op.name] = tf.zeros_like(var)

        # Iterate over the available batches to generate EWC samples
        # Note:  par['batch_size']//n should not be greater than ~150
        #        If this limit is reached, divide by a larger number and run
        #        more EWC batches to maintain both GPU memory requirements
        #        and network performance
        for i in range(par['batch_size']//par['EWC_batch_divisor']):

            # Compute gradients for each sample
            for grad, var in opt.compute_gradients(log_p_theta[i], var_list = self.variables_stabilization):

                # Aggregate gradients
                grad_dict[var.op.name] += tf.square(grad)/par['batch_size']/par['EWC_fisher_num_batches']

        # Iterate over the variables to assign values to the Fisher operations
        for var in self.variables_stabilization:
            fisher_ops.append(tf.assign_add(self.big_omega_var[var.op.name], grad_dict[var.op.name]))

        # Make update group
        self.update_big_omega = tf.group(*fisher_ops)


    def pathint_stabilization(self, adam_optimizer, prev_weights):
        """ Synaptic stabilization via the Zenke method """

        # Set up method
        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []

        # Iterate over variables in the model
        for var in self.variables_stabilization:

            # Reset the small omega vars and update the big omegas
            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append(tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0))
            update_big_omega_ops.append(tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(par['omega_xi'] + tf.square(var-prev_weights[var.op.name])))))

        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # Calculate the gradients and update the small omegas
        # This is called every batch
        with tf.control_dependencies([self.train_op]):
            self.delta_grads = adam_optimizer.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.task_loss, var_list = self.variables_stabilization)
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad))
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!


def main(save_fn, gpu_id=None):
    """ Run supervised learning training """

    # Update all dependencies in parameters
    update_dependencies()

    # Isolate requested GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # If desired, train the convolutional layers with the CIFAR datasets
    # Otherwise, the network will load convolutional weights from the saved file
    if (par['task'] == 'cifar' or par['task'] == 'imagenet') and par['train_convolutional_layers']:
        convolutional_layers.ConvolutionalLayers()

    print('\nRunning model.\n')

    # Reset TensorFlow graph
    tf.reset_default_graph()

    # Create placeholders for the model
    if par['task'] == 'mnist':
        x   = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][0]], 'stim')
    elif par['task'] == 'cifar' or par['task'] == 'imagenet':
        x   = tf.placeholder(tf.float32, [par['batch_size'], 32, 32, 3], 'stim')
    y       = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'out')
    mask    = tf.placeholder(tf.float32, [par['batch_size'], par['layer_dims'][-1]], 'mask')
    rule    = tf.placeholder(tf.float32, [par['batch_size'], par['n_tasks']], 'rulecue')
    gating  = [tf.placeholder(tf.float32, [par['layer_dims'][n+1]], 'gating') for n in range(par['n_layers']-1)]
    droput_keep_pct         = tf.placeholder(tf.float32, [], 'dropout')
    input_droput_keep_pct   = tf.placeholder(tf.float32, [], 'input_dropout')

    # Set up stimulus
    stim = stimulus.Stimulus(labels_per_task=par['labels_per_task'])

    # Initialize accuracy records
    accuracy_full = []
    accuracy_grid = np.zeros((par['n_tasks'], par['n_tasks']))

    # Enter TensorFlow session
    with tf.Session() as sess:

        # Select CPU or GPU
        device = '/cpu:0' if gpu_id is None else '/gpu:0'
        with tf.device(device):
            model = Model(x, y, gating, mask, droput_keep_pct, input_droput_keep_pct, rule)

        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(model.reset_prev_vars)

        # Begin training loop, iterating over tasks
        for task in range(par['n_tasks']):

            # Create dictionary of gating signals applied to each hidden layer for this task
            gating_dict = {k:v for k,v in zip(gating, par['gating'][task])}

            # Create rule cue vector for this task
            rule_cue = np.zeros([par['batch_size'], par['n_tasks']])
            rule_cue[:,task] = 1

            # Iterate over batches
            for i in range(par['n_train_batches']):

                # Make batch of training data
                stim_in, y_hat, mk = stim.make_batch(task, test = False)

                # Run the model using one of the available stabilization methods
                if par['stabilization'] == 'pathint':
                    _, _, loss, AL = sess.run([model.train_op, model.update_small_omega, model.task_loss, model.aux_loss], \
                        feed_dict={x:stim_in, y:y_hat, **gating_dict, mask:mk, droput_keep_pct:par['drop_keep_pct'], \
                        input_droput_keep_pct:par['input_drop_keep_pct'], rule:rule_cue})
                elif par['stabilization'] == 'EWC':
                    _, loss, AL = sess.run([model.train_op, model.task_loss, model.aux_loss], \
                        feed_dict={x:stim_in, y:y_hat, **gating_dict, mask:mk, droput_keep_pct:par['drop_keep_pct'], \
                        input_droput_keep_pct:par['input_drop_keep_pct'], rule:rule_cue})

                # Display network performance
                if i%500 == 0:
                    print('Iter: ', i, 'Loss: ', loss, 'Aux Loss: ',  AL)

            # Update big omegaes, and reset other values before starting new task
            if par['stabilization'] == 'pathint':
                sess.run(model.update_big_omega)
            elif par['stabilization'] == 'EWC':
                for _ in range(par['EWC_batch_divisor']*par['EWC_fisher_num_batches']):
                    stim_in, _, mk = stim.make_batch(task, test = False)
                    sess.run([model.update_big_omega], feed_dict = \
                        {x:stim_in, **gating_dict, mask:mk, droput_keep_pct:par['drop_keep_pct'], \
                        input_droput_keep_pct:par['input_drop_keep_pct'], rule:rule_cue})

            # Reset the Adam Optimizer, and set the prev_weight values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            if par['stabilization'] == 'pathint':
                sess.run(model.reset_small_omega)

            # Test the networks on all trained tasks
            num_test_reps = 10
            accuracy = np.zeros((task+1))
            for test_task in range(task+1):

                # Use appropriate gating and rule cues
                gating_dict = {k:v for k,v in zip(gating, par['gating'][test_task])}
                test_rule_cue = np.zeros([par['batch_size'], par['n_tasks']])
                test_rule_cue[:,test_task] = 1

                # Repeat the test as desired
                for r in range(num_test_reps):
                    stim_in, y_hat, mk = stim.make_batch(test_task, test = True)
                    acc = sess.run(model.accuracy, feed_dict={x:stim_in, y:y_hat, \
                        **gating_dict, mask:mk, droput_keep_pct:1.0, input_droput_keep_pct:1.0, rule:test_rule_cue})/num_test_reps
                    accuracy_grid[task, test_task]  += acc
                    accuracy[test_task] += acc

            # Display network performance after testing is complete
            print('Task ',task, ' Mean ', np.mean(accuracy), ' First ', accuracy[0], ' Last ', accuracy[-1])
            accuracy_full.append(np.mean(accuracy))

            # Reset weights between tasks if called upon
            if par['reset_weights']:
                sess.run(model.reset_weights)

        # Save model performance and parameters if desired
        if par['save_analysis']:
            save_results = {'task': task, 'accuracy': accuracy, 'accuracy_full': accuracy_full, \
                            'accuracy_grid': accuracy_grid, 'par': par}
            pickle.dump(save_results, open(par['save_dir'] + save_fn, 'wb'))

    print('\nModel execution complete.')
