### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import tensorflow as tf
from itertools import product

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

global par

par = {
    # General parameters
    'save_dir'              : './savedir/',
    'loss_function'         : 'cross_entropy',    # cross_entropy or MSE
    'stabilization'         : 'pathint', # 'EWC' (Kirkpatrick method) or 'pathint' (Zenke method)
    'learning_rate'         : 0.001,
    'task'                  : 'mnist',
    'save_analysis'         : True,
    'train_convolutional_layers' : False,

    # Task specs
    'n_tasks'               : 100,

    'layer_dims'            : [28**2, 2000, 2000, 10], # mnist
    #'layer_dims'            : [4096, 1000, 1000, 5], #cifar
    'pct_active_neurons'    : 1.0,
    'multihead'             : False, # option for CIFAR task, do we use different output neurons for each label nad add a mask, or recycle them

    # Dropout
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'conv_drop_keep_pct'    : 0.75,

    # Training specs
    'batch_size'            : 256,
    'n_train_batches'       : 3906, # 3906*256 = 20 epochs * 50000
    'n_batches_top_down'    : 10000,

    # Omega parameters
    'omega_c'               : 0.1,
    'omega_xi'              : 0.01,
    'last_layer_mult'       : 2,
    'scale_factor'          : 1,

    # Projection of top-down activity
    # Only one can be True
    'clamp'                 : None, # can be either 'dendrites', 'neurons', 'partial' or None
    'EWC_fisher_num_batches': 32, # number of batches size when calculating EWC

}

############################
### Dependent parameters ###
############################

def gen_gating():

    m = round(1/par['pct_active_neurons'])

    par['gating'] = []
    for t in range(par['n_tasks']):
        gating_task = []
        for n in range(par['n_layers']-2):
            gating_layer = np.zeros((par['layer_dims'][n+1]), dtype = np.float32)
            for i in range(par['layer_dims'][n+1]):
                if par['clamp'] == 'neurons':
                    if np.random.rand() < par['pct_active_neurons']:
                        gating_layer[i] = 1
                elif par['clamp'] == 'split':
                    if t%m == i%m:
                        if np.random.rand() < 0.5:
                            gating_layer[i] = 0.5
                        else:
                            gating_layer[i] = 1
                elif par['clamp'] == 'partial':
                    if np.random.rand() < 0.5:
                        gating_layer[i] = 0.5
                    else:
                        gating_layer[i] = 1
                elif par['clamp'] is None:
                    gating_layer[i] = 1
            gating_task.append(gating_layer)
        par['gating'].append(gating_task)


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    par['n_layers'] = len(par['layer_dims'])
    par['max_layer_dim'] = np.max(par['layer_dims'][1:-1])
    gen_gating()


def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    for (key, val) in updates.items():
        par[key] = val
        print('Updating : ', key, ' -> ', val)
    update_dependencies()


update_dependencies()
print("--> Parameters successfully loaded.\n")
