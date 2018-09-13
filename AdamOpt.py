import numpy as np
import tensorflow as tf

class AdamOpt:

    """
    Example of use:

    optimizer = AdamOpt.AdamOpt(variables, learning_rate=self.lr)
    self.train = optimizer.compute_gradients(self.loss, gate=0)
    gvs = optimizer.return_gradients()
    self.g = gvs[0][0]
    self.v = gvs[0][1]
    """

    def __init__(self, variables, learning_rate = 0.001):

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.t = 0
        self.variables = variables
        self.learning_rate = learning_rate

        self.m = {}
        self.v = {}
        self.delta_grads = {}
        for var in self.variables:
            self.m[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.v[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            self.delta_grads[var.op.name]  = tf.Variable(tf.zeros(var.get_shape()), trainable=False)

        self.grad_descent = tf.train.GradientDescentOptimizer(learning_rate = 1.0)


    def reset_params(self):

        self.t = 0
        reset_op = []
        for var in self.variables:
            reset_op.append(tf.assign(self.m[var.op.name], tf.zeros(var.get_shape())))
            reset_op.append(tf.assign(self.v[var.op.name], tf.zeros(var.get_shape())))
            reset_op.append(tf.assign(self.delta_grads[var.op.name], tf.zeros(var.get_shape())))

        return tf.group(*reset_op)


    def compute_gradients(self, loss, apply = True):

        self.gradients = self.grad_descent.compute_gradients(loss, var_list = self.variables)

        self.t += 1
        lr = self.learning_rate*np.sqrt(1-self.beta2**self.t)/(1-self.beta1**self.t)
        self.update_var_op = []

        for (grads, _), var in zip(self.gradients, self.variables):
            new_m = self.beta1*self.m[var.op.name] + (1-self.beta1)*grads
            new_v = self.beta2*self.v[var.op.name] + (1-self.beta2)*grads*grads

            delta_grad = - lr*new_m/(tf.sqrt(new_v) + self.epsilon)
            delta_grad = tf.clip_by_norm(delta_grad, 1)

            self.update_var_op.append(tf.assign(self.m[var.op.name], new_m))
            self.update_var_op.append(tf.assign(self.v[var.op.name], new_v))
            self.update_var_op.append(tf.assign(self.delta_grads[var.op.name], delta_grad))
            if apply:
                self.update_var_op.append(tf.assign_add(var, delta_grad))

        return tf.group(*self.update_var_op)


    def return_delta_grads(self):
        return self.delta_grads

    def return_means(self):
        return self.m

    def return_grads_and_vars(self):
        return self.gradients
