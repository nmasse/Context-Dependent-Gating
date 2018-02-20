import numpy as np
from parameters import *
import model
import sys, os
import pickle


def try_model(save_fn,gpu_id):

    try:
        # Run model
        model.main(save_fn, gpu_id)
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

imagenet_updates = {
    'layer_dims'            : [4096, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'imagenet',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 977,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }


# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


# updates for multi-head network, cifar only
multi_updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}

# updates for split networks
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10]}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5]}

print('ImageNet - Synaptic Stabilization = SI - Gating = 80%')
update_parameters(imagenet_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.80, 'input_drop_keep_pct': 1.0})
update_parameters({'stabilization': 'pathint', 'omega_c': 1.0, 'omega_xi': 0.01})
update_parameters({'train_convolutional_layers': True})
save_fn = 'imagenet_SI.pkl'
try_model(save_fn, gpu_id)
quit()


print('MNIST - Synaptic Stabilization = SI - Gating = 80%')
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'pathint', 'omega_c': 0.035, 'omega_xi': 0.01})
save_fn = 'mnist_SI.pkl'
try_model(save_fn, gpu_id)

print('MNIST - Synaptic Stabilization = EWC - Gating = 80%')
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'EWC', 'omega_c': 10})
save_fn = 'mnist_EWC.pkl'
try_model(save_fn, gpu_id)

print('CIFAR - Synaptic Stabilization = SI - Gating = 75%')
update_parameters(cifar_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.75, 'input_drop_keep_pct': 1.0})
update_parameters({'stabilization': 'pathint', 'omega_c': 0.2, 'omega_xi': 0.01})
update_parameters({'train_convolutional_layers': True})
save_fn = 'cifar_SI.pkl'
try_model(save_fn, gpu_id)

print('CIFAR - Synaptic Stabilization = EWC - Gating = 75%')
update_parameters(cifar_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.75, 'input_drop_keep_pct': 1.0})
update_parameters({'stabilization': 'EWC', 'omega_c': 10})
update_parameters({'train_convolutional_layers': False})
save_fn = 'cifar_EWC.pkl'
try_model(save_fn, gpu_id)
