import os
import numpy as np
import tensorflow.compat.v1 as tf
import AdamOpt


class Model:

    """ RNN model for supervised and reinforcement learning training """

    def __init__(self, input_data, target_data, lr, config):

        # Load input activity, target data, training mask, etc.
        self.input_data     = tf.unstack(input_data, axis=0)
        self.target_data    = tf.unstack(target_data, axis=0)
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


    def rnn_cell_loop(self):
        """ Initialize parameters and execute loop through
            time to generate the network outputs """

        # Specify training method outputs
        self.pol_out = []
        self.val_out = []
        self.action = []
        self.reward = []
        self.activity = []

        # intial LSTM activity
        with tf.variable_scope('initial_activity', reuse = tf.AUTO_REUSE):
            h = tf.get_variable('h', shape = (1, self.config['size_lstm']), dtype = tf.float32)
        h = tf.tile(h, (self.config['batch_size'], 1))
        c = tf.zeros((self.config['batch_size'], self.config['size_lstm']), dtype = tf.float32)

        # initial action
        action = tf.zeros((self.config['batch_size'], self.config['n_output']), dtype = tf.float32)
        # initial reward
        reward = tf.zeros((self.config['batch_size'], 1), dtype = tf.float32)

        # loop through the neural inputs, indexed in time
        for i, (input, target) in enumerate(zip(self.input_data, self.target_data)):

            x = input

            # create the low-rank weights using LSTM activity from the
            # previous time steps
            dynamic_W = self.create_low_rank_weights(h)

            for j in range(self.n_layers):

                x = self.dense(x, self.layer_dims[j+1], 'feedforward'+str(j), dynamic_W = dynamic_W[j])

                # the activity of the second to last layer will project
                # to the LSTM
                if j == self.n_layers - 2:
                    penultimate_layer = x

            lstm_input = tf.concat((penultimate_layer, action, reward), axis = -1)
            h, c = self.recurrent_cell(h, c, lstm_input, self.config['size_lstm'], 'LSTM')

            pol_out        = x
            val_out        = self.dense(h, 1, scope = 'value', activation = tf.identity)
            action_index   = tf.multinomial(pol_out, 1)
            action         = tf.one_hot(tf.squeeze(action_index), self.config['n_output'])
            pol_out        = tf.nn.softmax(pol_out, 1)  # Note softmax for entropy loss

            reward         = tf.reduce_sum(action*target, axis=1, keepdims=True)

            # Record RL outputs
            self.pol_out.append(pol_out)
            self.val_out.append(val_out)
            self.action.append(action)
            self.reward.append(reward)
            self.activity.append(h)


    def dynamic_layer(self, x, dynamic_W, n_output, scope, bias = True, activation = tf.nn.relu):

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

            W = tf.get_variable('W', shape = [x.shape[-1], n_output], dtype = tf.float32)
            b = tf.get_variable('b', shape = [1, n_output], initializer = tf.zeros_initializer(), \
                dtype = tf.float32) if bias else 0.

        W_effective = W * (1 + dynamic_W)

        return activation(tf.einsum('bi,bij->bj', x, W_effective) + b, name = 'output')



    def create_low_rank_weights(self, h):

        # split the neural activity, and project from each part to
        # different layers
        h_split = tf.reshape(h, (self.config['batch_size'], self.n_layers, -1))
        h_split = tf.unstack(h_split, axis = 1)

        dynamic_W = []

        for i in range(self.n_layers):
            if self.config['dynamic_weights_rank'][i] > 0:
                N = self.config['dynamic_weights_rank'][i]
                u =  self.dense(h_split[i], N*self.layer_dims[i], 'u'+str(i), activation = tf.identity)
                v =  self.dense(h_split[i], N*self.layer_dims[i+1], 'v'+str(i), activation = tf.identity)
                u  = tf.reshape(u,  (self.config['batch_size'], self.layer_dims[i], N))
                v  = tf.reshape(v,  (self.config['batch_size'], self.layer_dims[i+1], N))
                u  = tf.unstack(u, axis = -1)
                v  = tf.unstack(v, axis = -1)

                dW = tf.zeros((self.config['batch_size'], self.layer_dims[i], \
                    self.layer_dims[i+1]), dtype = tf.float32)

                for j in range(N):
                    dW += u[j][:, :, tf.newaxis] @ v[j][:, tf.newaxis, :]
                dynamic_W.append(dW)

            else:
                dynamic_W.append(None)

        return dynamic_W



    def recurrent_cell(self, h, c, rnn_input, lstm_size, scope_prefix):
        """ Using the appropriate recurrent cell
            architecture, compute the hidden state """

        u = tf.concat((rnn_input, h), axis = -1)

        f = tf.sigmoid(self.dense(u, lstm_size, scope_prefix + '_f', \
            activation = tf.identity))
        i = tf.sigmoid(self.dense(u, lstm_size, scope_prefix + '_i', \
            activation = tf.identity))
        cn = tf.tanh(self.dense(u, lstm_size, scope_prefix + '_i', \
            activation = tf.identity))
        c   = f * c + i * cn
        o = tf.sigmoid(self.dense(u, lstm_size, scope_prefix + '_o', \
            activation = tf.identity))
        h =  o * tf.tanh(c)

        return h, c


    def optimize(self):

        epsilon = 1e-6

        # Collect information from across time
        self.reward     = tf.stack(self.reward)
        self.action     = tf.stack(self.action)
        self.pol_out    = tf.stack(self.pol_out)

        # Get the value outputs of the network, and pad the last time step
        val_out = tf.concat((tf.stack(self.val_out), tf.zeros((1,self.config['batch_size'],1))), axis=0)


        # Compute predicted value and the advantage for plugging into the policy loss
        pred_val = self.reward + self.config['discount_rate']*val_out[1:,:,:]
        advantage = pred_val - val_out[:-1,:,:]

        # Stop gradients back through action, advantage, and mask
        action_static    = tf.stop_gradient(self.action)
        advantage_static = tf.stop_gradient(advantage)
        pred_val_static  = tf.stop_gradient(pred_val)


        # Policy loss
        self.pol_loss = -tf.reduce_mean(advantage_static*action_static*tf.log(epsilon+self.pol_out))

        # Value loss
        self.val_loss = 0.5*tf.reduce_mean(tf.square(val_out[:-1,:,:]-pred_val_static))

        # Entropy loss
        self.entropy_loss = -tf.reduce_mean(tf.reduce_sum(self.pol_out*tf.log(epsilon+self.pol_out), axis=2))

        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        #opt = AdamOpt.AdamOpt(variables = tf.trainable_variables(), learning_rate = self.learning_rate)

        self.train_op = opt.minimize(self.pol_loss + \
                                     self.config['value_cost']*self.val_loss - \
                                     self.config['entropy_cost']*self.entropy_loss)

        #self.train_op = opt.compute_gradients(self.pol_loss + \
        #                                      self.val_cost*self.val_loss - \
        #                                      self.entropy_cost*self.entropy_loss)



    def dense(self, x, n_output, scope, bias = True, activation = tf.nn.relu, dynamic_W = None):

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

            W = tf.get_variable('W', shape = [x.shape[-1], n_output], dtype = tf.float32)
            b = tf.get_variable('b', shape = [1, n_output], initializer = tf.zeros_initializer(), \
                dtype = tf.float32) if bias else 0.

        if dynamic_W is not None:

            # TODO: not sure of the best way to combine learned weights (W)
            # and dynamic weights
            # options:  W * (1 + dynamic_W),
            #           W * dynamic_W
            #           W + dynamic_W
            return activation(tf.einsum('bi,bij->bj', x, W * (1 + dynamic_W)) + b, name = 'output')
            #return activation(tf.einsum('bi,bij->bj', x, W + dynamic_W) + b, name = 'output')

        else:
            return activation(x @ W + b, name = 'output')
