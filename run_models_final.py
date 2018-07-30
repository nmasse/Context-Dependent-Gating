import numpy as np
from parameters import *
import model
import sys, os
import argparse
import pickle
from itertools import product

"""
Example Commands:

python run_models.py --task mnist --stab pathint --gate XdG --c 3 4 5 --xi 0 --versions 5 --gpu 0
python run_models.py --task imagenet --stab ewc --gate none --rule 1 --c 10 --versions 3 --gpu 1

"""

omegas_c_path = [0, 0.01, 0.015, 0.02, 0.035, 0.05, 0.1, 0.2]
omegas_c_EWC  = [0, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 2000, 4000, 8000]
omegas_xi     = [0.001, 0.01]


###############################################################################
### Parser ####################################################################
###############################################################################

class ParameterAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('')
        print(values)
        print(option_string[2:])
        update_parameters({option_string[2:]:values})



dyn_parser = argparse.ArgumentParser()
for key in par.keys():
    dyn_parser.add_argument('--'+key, action=ParameterAction)

args = dyn_parser.parse_args()
quit()

class ParameterAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('va', values)
        print(option_string)
        quit()

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, choices=['mnist','imagenet','cifar'], action=ParameterAction,\
    help='Task to run: MNIST, ImageNet, or CIFAR')
parser.add_argument('--stab', type=str, required=True, choices=['nostab', 'pathint', 'ewc'], \
    help='Stabilization to use: None, pathint (Zenke), or EWC (Kirkpatrick)')
parser.add_argument('--gate', type=str, required=True, choices=['nogate', 'split', 'XdG'], \
    help='Gating method to use: None, split network, or XdG')
parser.add_argument('--gatepct', type=float, required=False, default=0.8, \
    help='Percentage of units gated per task.')
parser.add_argument('--rule', type=int, required=False, default=0, choices=[0,1], \
    help='Whether to include rule cue: 0 = False, 1 = True')
parser.add_argument('--gpu', type=str, required=False, default=None, \
    help='GPU for running model.')

parser.add_argument('--c', type=int, nargs='+', required=False, default=[0], \
    help='List of C terms (by ID) to iterate over for stabilization method.')
parser.add_argument('--xi', type=int, nargs='+', required=False, default=[0.01], \
    help='List of Xi terms (by ID) to iterate over for stabilization method.')
parser.add_argument('--versions', type=int, required=False, default=1, \
    help='Number of times to run each value of C or Xi.')

parser.add_argument('--lr', type=float, required=False, default=0.001, \
    help='Learning rate.')
parser.add_argument('--trainconv', type=int, required=False, default=0, choices=[0,1], \
    help='Whether to train the convolutional layers for ImageNet/CIFAR: 0 = False, 1 = True')
parser.add_argument('--dropkeep', type=float, required=False, default=0.5, \
    help='Keep percentage for dropout in the hidden layers.')
parser.add_argument('--dropkeepinp', type=float, required=False, default=1., \
    help='Keep percentage for dropout in the network input.')
parser.add_argument('--dropkeepconv', type=float, required=False, default=0.75, \
    help='Keep percentage for dropout in the convolutional network.')
parser.add_argument('--multihead', type=int, required=False, default=0, choices=[0,1], \
    help='Whether the network output is multiheaded.')
parser.add_argument('--subnetworks', type=int, required=False, default=5, \
    help='Number of subnetworks for split gating.')

parser.add_argument('--ntasks', type=int, required=False, default=100, \
    help='Number of tasks to run, default is 100.')
parser.add_argument('--savedir', type=str, required=False, default='./savedir/', \
    help='Save directory.')
parser.add_argument('--batches', type=int, required=False, default=-1, \
    help='Number of batches to run of each task.')
parser.add_argument('--batchsize', type=int, required=False, default=256, \
    help='Number of trials per batch.')
parser.add_argument('--ewcbatches', type=int, required=False, default=-1, \
    help='Number of Fisher information sampling batches for EWC.')

args = parser.parse_args()
gpu_id = args.gpu
print('Using GPU {}.'.format(gpu_id))
updates = {}

updates['task'] = args.task
if updates['task'] in ['mnist', 'imagenet']:
    n_tasks = np.max([100, args.ntasks])
elif updates['task'] in ['cifar']:
    n_tasks = np.max([20, args.ntasks])

if n_tasks != args.ntasks:
    print('Reducing number of tasks to allowable maximum of {} for {}.'.format(n_tasks, updates['task']))
updates['n_tasks'] = n_tasks

updates['multihead'] = bool(args.multihead) if updates['task'] != 'mnist' else False
if bool(args.multihead) == True and updates['task'] == 'mnist':
    print('Multihead set to false due to MNIST task.')

updates['stabilization'] = None if args.stab.lower == 'nostab' else args.stab
updates['gating_type'] = None if args.gate.lower == 'nogate' else args.gate
updates['gate_pct'] = args.gatepct
updates['include_rule_signal'] = bool(args.rule)

updates['n_subnetworks'] = args.subnetworks
updates['train_convolutional_layers'] = bool(args.trainconv) if updates['task'] != 'mnist' else False
updates['batch_size'] = args.batchsize
updates['learning_rate'] = args.lr
updates['EWC_fisher_num_batches'] = 32 if args.ewcbatches < 0 else args.ewcbatches

updates['drop_keep_pct'] = args.dropkeep
updates['input_drop_keep_pct'] = args.dropkeepinp
updates['conv_drop_keep_pct'] = args.dropkeepconv

updates['save_dir'] = args.savedir

if args.task == 'mnist':
    updates['n_train_batches'] = 3906 if args.batches < 0 else args.batches
    updates['layer_dims'] = [784, 2000, 2000, 10] if updates['gating_type'] != 'split' else [784, 3665, 3665, 10]

elif args.task == 'imagenet':
    updates['n_train_batches'] = 977 if args.batches < 0 else args.batches
    updates['layer_dims'] = [4096, 2000, 2000, 10] if updates['gating_type'] != 'split' else [4096, 3665, 3665, 10]

elif args.task == 'cifar':
    updates['n_train_batches'] = 977 if args.batches < 0 else args.batches
    updates['layer_dims'] = [4096, 1000, 1000, 5] if updates['gating_type'] != 'split' else [4096, 1164, 1164, 5]

if updates['multihead']:
    updates['layer_dims'] *= updates['n_tasks']

if updates['gating_type'] == 'XdG' and updates['include_rule_signal'] == True:
    loop_key = False
    while key:
        x = input('Are you sure you want to have both XdG and rulecue? ' + \
                  'If (y), will proceed.  If (n), will disable rulecue.')
        if x.lower == 'y':
            loop_key = True
        elif x.lower == 'n':
            updates['include_rule_signal'] = False
            loop_key = True

if updates['stabilization'] == 'pathint':
    omegas_c = omegas_c_path
elif updates['stabilization'] == 'ewc':
    omegas_c = omegas_c_ewc

for v in range(args.versions):
    for xi_id, c_id in product(args.xi, args.c):

        print('\nUpdating and verifying parameters.')
        update_parameters(updates)
        updates['omega_c'] = omegas_c[c_id]
        updates['omega_xi'] = omegas_xi[xi_id]
        update_dependencies()

        rule = 'rulecue' if updates['include_rule_signal'] else 'norulecue'
        savefn = '_'.join([args.task, args.stab, args.gate, rule, 'c'+str(c_id), 'xi'+str(xi_id), 'v'+str(v)]) + '.pkl'
        print('\nFilename:', savefn)


quit('Success!')


def try_model(save_fn,gpu_id):
    """ Run the model unless asked to quit, in which case quit cleanly """

    try:
        model.main(save_fn, gpu_id)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')


###############################################################################
###############################################################################
###############################################################################


### Sets of task-specific parameters

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
    'n_train_batches'       : 977*1,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }


### Sets of multi-head network-specific parameters
multi_updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}
imagenet_multi_updates = {'layer_dims':[4096, 2000, 2000, 1000], 'multihead': True}


### Sets of split network-specific parameters
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10], 'multihead': False}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5], 'multihead': False}
imagenet_split_updates = {'layer_dims':[4096, 3665, 3665, 10], 'multihead': False}


# Standard omega_c values
omegas = [0.01, 0.015, 0.02, 0.035, 0.05, 0.1, 0.2]
omegas_EWC = [20,35,50,75,100,150,200,350,500,750,1000,2000, 4000,8000]


###############################################################################
###############################################################################
###############################################################################


def recurse_best(data_dir, prefix):
    """ Scan through already-completed models to test a model with parameters
        between those which have the highest accuracies """

    # Get filenames
    name_and_data = []
    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix):
            x = pickle.load(open(data_dir + full_fn, 'rb'))
            name_and_data.append((full_fn, x['accuracy_full'][-1], x['par']['omega_c']))

    # Find number of c's and v's
    cids = []
    vids = []
    for (f, _, _) in name_and_data:
        if f[-9].isdigit():
            c = f[-9:-7]
        else:
            c = f[-8]
        if c == 'R':
            # Don't look at existing files generated by this function
            print('Ignoring ', f)
            continue
        if c not in cids:
            cids.append(c)
        if f[-5] not in vids:
            vids.append(f[-5])

    # Show discovered data
    print(name_and_data)
    print(cids)
    print(vids)

    # Scan across c's and v's for accuracies
    accuracies = np.zeros((len(cids)))
    count = np.zeros((len(cids)))
    omegas = np.zeros((len(cids)))
    cids = sorted(cids)
    vids = sorted(vids)

    for (c_id, v_id) in product(range(len(cids)), range(len(vids))):
        text_c = 'omega'+str(cids[c_id])
        text_v = '_v'+str(vids[v_id])
        for full_fn in os.listdir(data_dir):
            if full_fn.startswith(prefix) and text_c in full_fn and text_v in full_fn:
                print('c_id', c_id)
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                accuracies[int(c_id)] += x['accuracy_full'][-1]
                count[int(c_id)] += 1
                omegas[int(c_id)] = x['par']['omega_c']

    # Show accuracies
    accuracies /= count
    print('accuracies ', accuracies)

    # Sort the     # Sort the accuracies andaccuracies and find the recursed parameter
    ind_sorted = np.argsort(accuracies)
    print('Sorted ind ', ind_sorted)
    if ind_sorted[-1] > ind_sorted[-2] or ind_sorted[-1] == len(ind_sorted)-1: # to the right
        cR = (omegas[ind_sorted[-1]] + omegas[ind_sorted[-1]-1])/2
    else:
        cR = (omegas[ind_sorted[-1]] + omegas[ind_sorted[-1]+1])/2

    # Show new parameters
    print('omegas ', omegas)
    print('cR = ', cR)

    # Get optimal parameters
    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix) and 'omega'+cids[ind_sorted[0]] in full_fn:
            opt_pars = pickle.load(open(data_dir + full_fn, 'rb'))['par']

    # Update parameters and run new model versions
    update_parameters(opt_pars)
    update_parameters({'omega_c' : cR})
    for i in range(5):
        save_fn = prefix + '_omegaR_v' + str(i) + '.pkl'
        print('save_fn', save_fn)
        print('save_dir', opt_pars['save_dir'])
        try_model(save_fn, sys.argv[1])
        print(save_fn, cR)


###############################################################################
###############################################################################
###############################################################################


def run_base():
    update_parameters(imagenet_updates)
    update_parameters({'gating_type': None,'gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_c': 0.0, 'omega_xi': 0.01})

    update_parameters({'reset_weights': True})
    for i in range(0,5):
        save_fn = 'imagenet_weight_reset_v' + str(i) + '.pkl'
        #save_fn = 'imagenet_base_omega0_v' + str(i) + '.pkl'
        try_model(save_fn, gpu_id)

    update_parameters(imagenet_multi_updates)
    for i in range(1,0):
        save_fn = 'imagenet_baseMH_v' + str(i) + '.pkl'
        try_model(save_fn, gpu_id)

def run_with_rule():

    omegas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    update_parameters(mnist_updates)
    update_parameters({'gating_type':None, 'gate_pct':0.80, 'input_drop_keep_pct':1.0, \
        'stabilization':'pathint', 'omega_xi':0.01, 'include_rule_signal':True})

    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c':omegas[j]})
            save_fn = 'mnist_SI_rulecue_nogate_omega'+str(j)+'_v'+str(i)+'.pkl'
            try_model(save_fn, gpu_id)

    update_parameters({'stabilization':'EWC'})

    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c':omegas[j]})
            save_fn = 'mnist_EWC_rulecue_nogate_omega'+str(j)+'_v'+str(i)+'.pkl'
            try_model(save_fn, gpu_id)



def run_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': None,'gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(1,0):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

    update_parameters(imagenet_multi_updates)
    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_MH_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_partial_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'partial','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(1,5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_partial_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_XdG_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(0,1):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_XdG_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_split_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters(imagenet_split_updates)
    update_parameters({'gating_type': 'split','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(1,len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_split_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_split_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200]

    update_parameters(imagenet_updates)
    update_parameters(imagenet_split_updates)
    update_parameters({'gating_type': 'split','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(0,1):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_split_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_partial_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'partial','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'EWC', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_partial_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_XdG_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'EWC', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_XdG_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)


def run_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': None,'gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'EWC', 'omega_xi': 0.01})

    for i in range(1,0):
        for j in range(len(omegas)-1, len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

    update_parameters(imagenet_multi_updates)
    for i in range(5):
        for j in range(len(omegas)-1, len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_MH_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_EWC_split')
#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_SI_split')
#run_EWC()
#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_EWC_omega')
#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_SI_XdG')

#run_EWC()
#run_base()
#run_SI()
#run_split_EWC()
#run_split_SI()
#run_partial_SI()
#run_XdG_SI()
#run_partial_EWC()
#run_XdG_EWC()
run_with_rule()

quit()




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
