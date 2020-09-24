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

        # intial LSTM anactivity
        h = tf.zeros((self.config['batch_size'], self.config['size_lstm']), dtype = tf.float32)
        c = tf.zeros((self.config['batch_size'], self.config['size_lstm']), dtype = tf.float32)

        # initial action
        action = tf.zeros((self.config['batch_size'], self.config['n_output']), dtype = tf.float32)
        # initial reward
        reward = tf.zeros((self.config['batch_size'], 1), dtype = tf.float32)


        # Loop through the neural inputs, indexed in time
        for i, (input, target) in enumerate(zip(self.input_data, self.target_data)):

            x = input
            for j, d in enumerate(self.config['size_feedforward']):

                if j < len(self.config['size_feedforward']) - 1:
                    x = self.dense(x, d, 'feedforward'+str(j))
                else:
                    x = tf.nn.relu(tf.einsum('bi,bij->bj', x, W_effective) + b)


                # use the output of the second layer as input into the LSTM
                if j == 1:
                    lstm_input = tf.concat((x, 3*action, 3*reward), axis = -1)
                    h, c = self.recurrent_cell(h, c, lstm_input, self.config['size_lstm'], 'LSTM')

                # generate the low rank matrix
                # not sure if this should be done before or after the LSTM loop is run
                u =  self.dense(h, 2*self.config['size_feedforward'][-2], 'u', activation = tf.identity)
                v =  self.dense(h, 2*self.config['size_feedforward'][-1], 'v', activation = tf.identity)
                u  = tf.reshape(u,  (self.config['batch_size'], self.config['size_feedforward'][-2], 2))
                v  = tf.reshape(v,  (self.config['batch_size'], self.config['size_feedforward'][-1], 2))
                u  = tf.unstack(u, axis = -1)
                v  = tf.unstack(v, axis = -1)

                # calculate rank 2 dynamic matrix
                dynamic_W = u[0][:, :, tf.newaxis] @ v[0][:, tf.newaxis, :] + \
                    u[1][:, :, tf.newaxis] @ v[1][:, tf.newaxis, :]


                # weights to the output layer will consist of standard trainable weights,
                # multiplied with rank-2 dynamic weights calculated above
                with tf.variable_scope('penultimate_layer', reuse = tf.AUTO_REUSE):

                    W = tf.get_variable('W', shape = (self.config['size_feedforward'][-1], \
                        self.config['size_feedforward'][-2]), dtype = tf.float32)
                    b = tf.get_variable('b', shape = (1, self.config['size_feedforward'][-1]), \
                        initializer = tf.zeros_initializer(), dtype = tf.float32)

                    W_effective = W * (1 + dynamic_W)


            #pol_out = tf.einsum('bi,ij->bj', x, W_pol) + b_pol
            pol_out        = self.dense(h, self.config['n_output'], scope = 'policy', activation = tf.identity)

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



    def dense(self, x, n_output, scope, bias = True, activation = tf.nn.relu):

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):

            W = tf.get_variable('W', shape = [x.shape[-1], n_output], dtype = tf.float32)
            b = tf.get_variable('b', shape = [1, n_output], initializer = tf.zeros_initializer(), \
                dtype = tf.float32) if bias else 0.

        return activation(x @ W + b, name = 'output')
