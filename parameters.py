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
    'task'                  : 'cifar',
    'save_analysis'         : True,
    'train_convolutional_layers' : False,

    # Task specs
    'n_tasks'               : 100,


    # Network shape
    'n_td'                  : 100,
    'n_dendrites'           : 1,
    #'layer_dims'            : [28**2, 100, 100, 10], # mnist
    'layer_dims'            : [4096, 500, 500, 100], #cifar
    'dendrites_final_layer' : False,
    'pct_active_neurons'    : 1.0,
    'multihead'             : False, # option for CIFAR task, do we use different output neurons for each label nad add a mask, or recycle them

    # Dropout
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 0.8,

    # Training specs
    'batch_size'            : 256,
    'n_train_batches'       : 3906, # 3906*256 = 20 epochs * 50000
    'n_batches_top_down'    : 15000,

    # Omega parameters
    'omega_c'               : 0.05*,
    'omega_xi'              : 0.1,


    # Projection of top-down activity
    # Only one can be True
    'clamp'                 : None, # can be either 'dendrites', 'neurons', 'partial' or None

    'EWC_fisher_calc_batch' : 128, # batch size when calculating EWC
    'EWC_fisher_num_batches': 64, # number of batches size when calculating EWC

}

############################
### Dependent parameters ###
############################

def gen_td_cases():

    # will create par['n_tasks'] number of tunings, each with exactly n non-zero elements equal to one
    # the distance between all tuned will be d

    par['td_cases'] = np.zeros((par['n_tasks'], par['n_td']), dtype = np.float32)

    #if par['clamp'] == 'neurons':
    for n in range(par['n_tasks']):
        par['td_cases'][n, n%par['n_td']] = 1


def gen_td_targets():

    m = round(1/par['pct_active_neurons'])
    print('Clamping: selecting every ', m, ' neuron')

    par['td_targets'] = []
    par['W_td0'] = []
    for n in range(par['n_layers']-1):
        td = np.zeros((par['n_tasks'],par['n_dendrites'], par['layer_dims'][n+1]), dtype = np.float32)
        Wtd = np.zeros((par['n_td'],par['n_dendrites'], par['layer_dims'][n+1]), dtype = np.float32)
        for i in range(par['layer_dims'][n+1]):

            if par['clamp'] == 'dendrites':
                for t in range(0, par['n_tasks'], par['n_dendrites']):
                    q = np.random.permutation(par['n_dendrites'])
                    for j, d in enumerate(q):
                        if t+j<par['n_tasks']:
                            td[t+j,d,i] = 1
                            Wtd[t+j,d,i] = 1

            elif par['clamp'] == 'neurons':
                for t in range(0, par['n_tasks']):
                    if np.random.rand() < 1/m:
                        td[t,:,i] = 1
                        Wtd[t,:,i] = 1

            elif par['clamp'] == 'split':
                for t in range(0, par['n_tasks']):
                    if t%m == i%m:
                        if np.random.rand(1) < 0.5:
                            td[t,:,i] = 0.5
                            Wtd[t,:,i] = 0.5
                        else:
                            td[t,:,i] = 1
                            Wtd[t,:,i] = 1

            elif par['clamp'] == 'partial':
                for t in range(0, par['n_tasks']):
                    if np.random.rand(1) < 0.5:
                        td[t,:,i] = 0.5
                        Wtd[t,:,i] = 0.5
                    else:
                        td[t,:,i] = 1
                        Wtd[t,:,i] = 1

            elif par['clamp'] is None:
                td[:,:,:] = 1
                Wtd[:,:,:] = 1

        par['td_targets'].append(td)
        par['W_td0'].append(Wtd)


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    par['n_layers'] = len(par['layer_dims'])
    gen_td_cases()
    gen_td_targets()


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
