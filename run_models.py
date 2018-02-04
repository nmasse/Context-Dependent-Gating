import numpy as np
from parameters import *
import model
import sys, os
import pickle


def try_model(save_fn,gpu_id):
    # GPU designated by first argument (must be integer 0-3)
    try:
        print('Selecting GPU ',  sys.argv[1])
        assert(int(sys.argv[1]) in [0,1,2,3])
    except AssertionError:
        quit('Error: Select a valid GPU number.')

    try:
        # Run model
        model.main(save_fn, sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

###############################################################################
###############################################################################
###############################################################################


mnist_updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'mnist',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 3906,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

cifar_updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'n_tasks'               : 20,
    'task'                  : 'cifar',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 977,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }

# updates for multi-head network
multi_updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}

# updates for split networks
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10], 'batch_size':128, \
    'n_train_batches': 3906*2,'EWC_fisher_num_batches': 64}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5]}


print('MNIST - Synaptic Stabilization = SI - Gating = 80%')
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'pathint', 'omega_c': 0.035, 'omega_xi': 0.01})
save_fn = 'mnist_SI.pkl'
try_model(save_fn, sys.argv[1])

print('MNIST - Synaptic Stabilization = EWC - Gating = 80%')
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'EWC', 'omega_c': 30})
save_fn = 'mnist_EWC.pkl'
try_model(save_fn, sys.argv[1])
