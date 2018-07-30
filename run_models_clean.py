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
    'input_drop_keep_pct'   : 1.,
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
    'n_train_batches'       : 1954,
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
multi_updates = {'layer_dims':[4096, 2000, 2000, 1000], 'multihead': True}

# updates for split networks
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10]}
imagenet_split_updates = {'layer_dims':[4096, 3665, 3665, 10]}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5]}

omegas = [0.01, 0.015, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5,0.75, 1  ]
omegas_imagenet = [0.05, 0.075, 0.1, 0.15, 0.2, 0.5,0.75, 1, 1.5, 2  ]
omegas_EWC = [10,20,35,50, 75, 100,150,200,350,500,750,1000,1500, 2000, 3500, 4000,8000]
omegas_EWC_XdG = [2,3.5, 5,7.5, 10,20,35,50, 75, 100,150,200,350,500,750,1000,1500, 2000, 3500, 4000,8000]

"""

savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/SI XdG/'
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG', 'gate_pct': 0.8, 'input_drop_keep_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'n_train_batches': 3906*5})
for oc in [0]:
    update_parameters({ 'omega_c': omegas[oc]/2})
    for v in range(0,5):
        save_fn = 'mnist_SI_XdG_100ep_omegaR' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
quit()
"""


"""
print('MNIST - Synaptic Stabilization = None - Gating = 80%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/'
savedir = './savedir/'
update_parameters(mnist_updates)
#update_parameters(mnist_split_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100,'n_train_batches': 1*3906, 'learning_rate': 1e-3})
#for oc in [2]:
update_parameters({ 'omega_c': 0.})
for v in range(0,5):
    save_fn = 'mnist_nostab_XdG_v' + str(v) + '.pkl'
    try_model(save_fn, gpu_id)
quit()
"""


print('ImageNet - Synaptic Stabilization = None - Gating = 80%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/'
savedir = './savedir/'
update_parameters(imagenet_updates)
#update_parameters(mnist_split_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
#for oc in [2]:
update_parameters({ 'omega_c': 0.})
for v in range(0,5):
    save_fn = 'imagenet_nostab_XdG_v' + str(v) + '.pkl'
    try_model(save_fn, gpu_id)
quit()

"""

print('ImageNet - Synaptic Stabilization = EWC - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/EWC XdG/'
update_parameters(imagenet_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'EWC', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
update_parameters({'train_convolutional_layers':False})
for v in range(0,5):
    for oc in [3,2]:
        update_parameters({ 'omega_c': omegas_EWC_XdG[oc]})
        save_fn = 'ImageNet_EWC_XdG_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
        update_parameters({'train_convolutional_layers':False})
quit()

"""

print('ImageNet - Synaptic Stabilization = EWC - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/SI/'
update_parameters(imagenet_updates)
#update_parameters(multi_updates)
update_parameters({'gating_type': None,'gate_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
update_parameters({'train_convolutional_layers':False})
for v in range(3,5):
    for oc in [3,4,5,6,7]:
        update_parameters({ 'omega_c': omegas[oc]})
        save_fn = 'ImageNet_SI_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
        update_parameters({'train_convolutional_layers':False})
quit()

print('ImageNet - Synaptic Stabilization = EWC - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/EWC/'
update_parameters(imagenet_updates)
update_parameters({'gating_type': None,'gate_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'EWC', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
update_parameters({'train_convolutional_layers':False})
for v in range(0,5):
    for oc in [5,6,7]:
        update_parameters({ 'omega_c': omegas_EWC_XdG[oc]})
        save_fn = 'ImageNet_EWC_XdG_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
        update_parameters({'train_convolutional_layers':False})
quit()

print('ImageNet - Synaptic Stabilization = EWC - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/EWC MH/'
update_parameters(imagenet_updates)
update_parameters(multi_updates)
update_parameters({'gating_type': None,'gate_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'EWC', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
update_parameters({'train_convolutional_layers':False})
for v in range(0,5):
    for oc in [2,3,4]:
        update_parameters({ 'omega_c': omegas_EWC_XdG[oc]})
        save_fn = 'ImageNet_EWC_XdG_MH_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
        update_parameters({'train_convolutional_layers':False})
quit()

print('ImageNet - Synaptic Stabilization = EWC - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/SI XdG/'
update_parameters(imagenet_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'include_rule_signal': False})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
update_parameters({'train_convolutional_layers':False})
for v in range(4,5):
    for oc in [7,8]:
        update_parameters({ 'omega_c': omegas[oc]})
        save_fn = 'ImageNet_SI_XdG_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
        update_parameters({'train_convolutional_layers':False})
quit()
"""


print('ImageNet - Synaptic Stabilization = EWC - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/EWC Rule Split/'
update_parameters(imagenet_updates)
update_parameters(imagenet_split_updates)
update_parameters({'gating_type': 'split','gate_pct': 0.8, 'include_rule_signal': True})
update_parameters({'stabilization': 'EWC', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
update_parameters({'train_convolutional_layers':False})
for v in range(1,5):
    for oc in [3]:
        update_parameters({ 'omega_c': omegas_EWC[oc]})
        save_fn = 'ImageNet_EWC_rule_split_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
        update_parameters({'train_convolutional_layers':False})
quit()

"""
print('ImageNet - Synaptic Stabilization = EWC - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/SI Rule Split/'
update_parameters(imagenet_updates)
update_parameters(imagenet_split_updates)
update_parameters({'gating_type': 'split','gate_pct': 0.8, 'include_rule_signal': True})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
update_parameters({'train_convolutional_layers':False})
for v in range(0,5):
    for oc in [7]:
        update_parameters({ 'omega_c': omegas_imagenet[oc]})
        save_fn = 'ImageNet_SI_rule_split_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
        update_parameters({'train_convolutional_layers':False})
quit()
"""






print('ImageNet - Synaptic Stabilization = SI - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/SI Rule/'
update_parameters(imagenet_updates)
update_parameters({'gating_type': None,'gate_pct': 0.0, 'include_rule_signal': True})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
for v in range(0,5):
    for oc in [7]:
        update_parameters({ 'omega_c': omegas_imagenet[oc]})
        save_fn = 'ImageNet_SI_rule_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
quit()

"""

print('ImageNet - Synaptic Stabilization = SI - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/EWC Rule/'
update_parameters(imagenet_updates)
update_parameters({'gating_type': None,'gate_pct': 0.0, 'include_rule_signal': True})
update_parameters({'stabilization': 'EWC', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100, 'learning_rate': 1e-3})
for v in range(1,5):
    for oc in [3]:
        update_parameters({ 'omega_c': omegas_EWC[oc]})
        save_fn = 'ImageNet_EWC_rule_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
quit()


"""

print('MNIST - Split - Synaptic Stabilization = EWC - Rule Cue - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/EWC Rule Split/'
update_parameters(mnist_updates)
update_parameters(mnist_split_updates)
#update_parameters({'layer_dims':[784, 3665, 3665, 10]}) #mnist_split_updates)
update_parameters({'gating_type': 'split','gate_pct': 0.8, 'input_drop_keep_pct': 0.8, 'include_rule_signal': True})
update_parameters({'stabilization': 'EWC', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100})
for oc in [8]:
    update_parameters({ 'omega_c': omegas_EWC[oc]})
    for v in range(0,5):
        save_fn = 'mnist_EWC_rule_split_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
quit()


print('MNIST - Synaptic Stabilization = EWC - Rule Cue - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/EWC Rule/'
update_parameters(mnist_updates)
update_parameters({'gating_type': None,'gate_pct': 0.8, 'input_drop_keep_pct': 0.8, 'include_rule_signal': True})
update_parameters({'stabilization': 'EWC', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100})
for oc in [0,1,2,3]:
    update_parameters({ 'omega_c': omegas_EWC[oc]})
    for v in range(3,5):
        save_fn = 'mnist_EWC_rule_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
quit()
"""

print('MNIST - Synaptic Stabilization = SI - Rule Cue - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/SI Rule/'
update_parameters(mnist_updates)
update_parameters({'gating_type': None, 'gate_pct': 0.0, 'input_drop_keep_pct': 0.8, 'include_rule_signal': True})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100})
for oc in [5]:
    update_parameters({ 'omega_c': omegas[oc]})
    for v in range(3,5):
        save_fn = 'mnist_SI_rule_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
quit()



print('MNIST - Split - Synaptic Stabilization = SI - Rule Cue - Gating = 0%')
savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/SI Rule Split/'
update_parameters(mnist_updates)
update_parameters(mnist_split_updates)
update_parameters({'gating_type': 'split','gate_pct': 0.8, 'input_drop_keep_pct': 0.8, 'include_rule_signal': True})
update_parameters({'stabilization': 'pathint', 'omega_xi': 0.01, 'save_dir': savedir, 'n_tasks': 100})
for oc in [0,1,2]:
    update_parameters({ 'omega_c': omegas[oc]})
    for v in range(0,5):
        save_fn = 'mnist_SI_split_rule_omega' + str(oc) + '_v' + str(v) + '.pkl'
        try_model(save_fn, gpu_id)
quit()


print('MNIST - Synaptic Stabilization = SI - Rule Cue - Gating = 0%')
update_parameters(mnist_updates)
update_parameters({'gating_type': None,'gate_pct': 0.0, 'input_drop_keep_pct': 0.8, 'include_rule_signal': True})
update_parameters({'stabilization': 'pathint', 'omega_c': 0.1, 'omega_xi': 0.01})
save_fn = 'mnist_SI_rule_oc1.pkl'
try_model(save_fn, gpu_id)
quit()



"""
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
"""

print('MNIST - Synaptic Stabilization = EWC - Gating = 0%')
update_parameters(mnist_updates)
update_parameters({'gating_type': None,'gate_pct': 0.0, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'EWC', 'omega_c': 1000}) # used to be 400
save_fn = 'test_mnist_EWC.pkl'
try_model(save_fn, gpu_id)
quit()

print('MNIST - Synaptic Stabilization = EWC - Gating = 80%')
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'EWC', 'omega_c': 10})
save_fn = 'test_mnist_EWC.pkl'
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
