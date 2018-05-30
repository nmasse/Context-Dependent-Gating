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
    'learning_rate'         : 0.004,
    'task'                  : 'mnist',
    'save_analysis'         : True,
    'train_convolutional_layers' : False,
    'reset_weights'         : False, # reset weights between tasks

    # Task specs
    'n_tasks'               : 100,

    'layer_dims'            : [28**2, 2000, 2000, 10], # mnist
    #'layer_dims'            : [4096, 1000, 1000, 5], #cifar
    'gate_pct'              : 0.0, # percentage of hidden units to gate. Only used when gating_type is set to XdG
    'n_subnetworks'         : 5, # Only used when gating_type is set to split
    'multihead'             : False, # option for CIFAR task, in which different unique output units are asscoaited with each label
    'gate_cost'             : np.array([1.,0.1]),

    # Dropout
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'conv_drop_keep_pct'    : 0.75,

    # Training specs
    'batch_size'            : 256,
    'n_train_batches'       : 3906, # 3906*256 = 20 epochs * 50000
    'n_batches_top_down'    : 20000,

    # Omega parameters
    'omega_c'               : 0.1,
    'omega_xi'              : 0.01,

    'EWC_fisher_num_batches': 16, # was 16, number of batches size when calculating EWC

    # Type of gating signal
    'gating_type'           : None, # can be either 'XdG', 'partial', 'split' or None

}

############################
### Dependent parameters ###
############################

def gen_gating():
    """
    Generate the gating signal to applied to all hidden units
    """
    par['gating'] = []
    for t in range(par['n_tasks']):
        gating_task = []
        for n in range(par['n_layers']-2):
            gating_layer = np.zeros((par['layer_dims'][n+1]), dtype = np.float32)
            for i in range(par['layer_dims'][n+1]):
                if par['gating_type'] == 'XdG':
                    if np.random.rand() < 1-par['gate_pct']:
                        gating_layer[i] = 1
                elif par['gating_type'] == 'split':
                    if t%par['n_subnetworks'] == i%par['n_subnetworks']:
                        if np.random.rand() < 0.5:
                            gating_layer[i] = 0.5
                        else:
                            gating_layer[i] = 1
                elif par['gating_type'] == 'partial':
                    if np.random.rand() < 0.5:
                        gating_layer[i] = 0.5
                    else:
                        gating_layer[i] = 1
                elif par['gating_type'] is None:
                    gating_layer[i] = 1
            gating_task.append(gating_layer)
        par['gating'].append(gating_task)



def update_dependencies():
    """
    Updates all parameter dependencies
    """

    par['n_layers'] = len(par['layer_dims'])
    if par['task'] == 'mnist' or par['task'] == 'imagenet':
        par['labels_per_task'] = 10
    elif par['task'] == 'cifar':
        par['labels_per_task'] = 5
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
