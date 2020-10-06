import os
import numpy as np
import tensorflow.compat.v1 as tf
import AdamOpt


class Model:

    """ RNN model for supervised and reinforcement learning training """

    def __init__(self, input_data, target_data, context_signal, lr, config):

        # Load input activity, target data, training mask, etc.
        self.input_data     = input_data[0, ...]
        self.target_data    = target_data[0, ...]
        self.context_signal  = context_signal
        self.learning_rate  = lr

        self.config         = config
        self.layer_dims     = [self.config['n_input']] + \
                              self.config['size_feedforward'] + \
                              [self.config['n_output']]
        self.n_layers       = len(self.layer_dims) - 1

        # Build the Tensorflow graph
        self.rnn_cell_loop()

        # Train the model
        self.optimize()

        self.count_number_variables()


    def count_number_variables(self):

        n = 0
        for v in tf.trainable_variables():
            n += np.prod(v.shape)

        print('NUMBER of VARS',n)


    def rnn_cell_loop(self):
        """ Initialize parameters and execute loop through
            time to generate the network outputs """




        if self.config['bias_or_dyn'] == 'dyn':
            context = None
        else:
            context = self.context_signal


        x = self.input_data
        for j in range(self.n_layers):

            print('LAYER', j)



            activation = tf.nn.relu if j < self.n_layers - 1 else tf.identity
            #x_concat = tf.concat((x, self.context_signal), axis = -1)
            x_concat = tf.concat((x), axis = -1)

            if j == 2:
                #self.activity = x
                if self.config['bias_or_dyn'] == 'dyn':
                    dynamic_W = self.create_dynamic_weights(self.context_signal, j)

                else:
                    dynamic_W = None

            else:
                dynamic_W = None

            #x = tf.nn.dropout(x, 0.5)

            x = self.dense(x,
                           self.layer_dims[j+1],
                           'feedforward'+str(j),
                           activation = activation,
                           dynamic_W = dynamic_W,
                           context = context,
                           bias = False)
            if j == 2:
                self.activity = x


        self.output = x


        action_index   = tf.multinomial(x, 1)
        action         = tf.one_hot(tf.squeeze(action_index), self.config['n_output'])
        self.reward    = tf.reduce_sum(action*self.target_data, axis=1, keepdims=True)

    def create_dynamic_weights(self, x, layer_num):
        N = 2
        u =  self.dense(x, N*self.layer_dims[layer_num], 'dyn_u'+str(layer_num), activation = tf.identity)
        v =  self.dense(x, N*self.layer_dims[layer_num+1], 'dyn_v'+str(layer_num), activation = tf.identity)
        u  = tf.reshape(u,  (self.config['batch_size'], self.layer_dims[layer_num], N))
        v  = tf.reshape(v,  (self.config['batch_size'], self.layer_dims[layer_num+1], N))
        u  = tf.unstack(u, axis = -1)
        v  = tf.unstack(v, axis = -1)

        dW = tf.zeros((self.config['batch_size'], self.layer_dims[layer_num], \
            self.layer_dims[layer_num+1]), dtype = tf.float32)

        for j in range(N):
            dW += u[j][:, :, tf.newaxis] @ v[j][:, tf.newaxis, :]

        return dW



    def optimize(self):


        context_vars = []
        for var in tf.trainable_variables():
            if 'dyn_' in var.op.name or 'context_' in var.op.name:
                 context_vars.append(var)

        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        #opt = AdamOpt.AdamOpt(variables = tf.trainable_variables(), learning_rate = self.learning_rate)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.target_data, \
                                                                              logits = self.output))

        all_gvs = opt.compute_gradients(self.loss, var_list = tf.trainable_variables())
        ctx_gvs = opt.compute_gradients(self.loss, var_list = context_vars)
        self.train_op = opt.apply_gradients(all_gvs)
        self.train_op_ctx = opt.apply_gradients(ctx_gvs)


    def dense(self, x, n_output, scope, bias = True, activation = tf.nn.relu, dynamic_W = None, context = None):

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

            W = tf.get_variable('W', shape = [x.shape[-1], n_output], dtype = tf.float32)
            b = tf.get_variable('b', shape = [1, n_output], initializer = tf.zeros_initializer(), \
                dtype = tf.float32) if bias else 0.

        if context is not None:

            with tf.variable_scope('context_' + scope, reuse = tf.AUTO_REUSE):

                Wc = tf.get_variable('context_W', shape = [context.shape[-1], n_output], dtype = tf.float32)
                #bc = tf.get_variable('context_b', shape = [1, n_output], initializer = tf.zeros_initializer(), \
                #    dtype = tf.float32) if bias else 0.
                print('context', context)
                print('Wc', Wc)
                bias = tf.einsum('bi,ij->bj', context, Wc) #+ bc
                #bias = bias[tf.newaxis, :]

        else:
            bias = 0.

        if dynamic_W is not None:

            # TODO: not sure of the best way to combine learned weights (W)
            # and dynamic weights
            # options:  W * (1 + dynamic_W),
            #           W * dynamic_W
            #           W + dynamic_W
            #return activation(tf.einsum('bi,bij->bj', x, W * (1 + dynamic_W)) + b, name = 'output')
            print('x', x)
            print('b', b)
            print('W', W)
            print('dynamic_W', dynamic_W)
            #return activation(tf.einsum('bi,bij->bj', x, W * (1+dynamic_W)) + b, name = 'output')
            W_eff = W + dynamic_W

            return activation(tf.einsum('bi,bij->bj', x, W_eff) + b + bias, name = 'output')
        else:
            print('x', x)
            print('W', W)
            print('b', b)
            print('bias', bias)
            return activation(x @ W + b + bias, name = 'output')
