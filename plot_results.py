import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
from itertools import product

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "arial"

def plot_fig2_new():

    #savedir = './savedir/mnist/'
    savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/'
    savedir1 = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/'
    all_same_scale = True
    f, ax1 = plt.subplots(1,1,figsize=(4,2.5))
    accuracy = {}

    ylim_min = 0.6

    # Figure 2A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    base    = 'mnist_no_stabilization'
    EWC     = 'mnist_EWC'
    SI      = 'mnist_SI'
    XdG      = 'mnist_nostab'
    #SI= 'neurons_InpDO_mnist_SI'
    b1 = plot_best_result(ax1, savedir + 'Baseline/', base, col=[0,0,0], label='Base')
    b2 = plot_best_result(ax1, savedir + 'EWC Fixed/', EWC, col=[0,1,0], label='EWC')
    b3 = plot_best_result(ax1, savedir + 'SI/', SI, col=[1,0,1], label='SI')
    b4 = plot_best_result(ax1, savedir + 'XdG/', XdG, col=[0,0,0], label='XdG',linestyle='--')


    SI_TD    = 'neurons_InpDO_mnist_SI'
    EWC_TD   = 'neurons_InpDO_mnist_EWC'
    b5 = plot_best_result(ax1, savedir + 'SI XdG/', SI_TD, col=[1,0,1], label='SI+XdG',linestyle='--')
    b6 = plot_best_result(ax1, savedir + 'EWC XdG/', EWC_TD, col=[0,1,0], label='EWC+XdG',linestyle='--')
    accuracy['base'] = b1
    accuracy['EWC'] = b2
    accuracy['SI'] = b3
    accuracy['XdG'] = b4
    accuracy['SI_XdG'] = b5
    accuracy['EWC_XdG'] = b6


    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [ylim_min,1],[0,100],[])

    plt.tight_layout()
    plt.savefig('Figure2.pdf', format='pdf')
    plt.show()

    f, (ax1, ax2) = plt.subplots(2,1,figsize=(4,6))
    ylim_min = 0.85

    b5 = plot_best_result(ax1, savedir + 'SI XdG/', SI_TD, col=[1,0,1], label='SI+XdG',linestyle='--')
    b6 = plot_best_result(ax1, savedir + 'EWC XdG/', EWC_TD, col=[0,1,0], label='EWC+XdG',linestyle='--')

    SI_TDP  = 'mnist_SI_rule'
    EWC_TDP = 'mnist_EWC_rule'
    b1 = plot_best_result(ax1, savedir + 'SI Rule/', SI_TDP, col=[1,0,1], label='SI+Partial')
    b3 = plot_best_result(ax1, savedir + 'EWC Rule/', EWC_TDP, col=[0,1,0], label='EWC+Partial')

    accuracy['SI_Rule'] = b1
    accuracy['EWC_Rule'] = b3

    SI_TDPS  = 'mnist_SI_split_rule'
    EWC_TDPS = 'mnist_EWC_rule_split'

    b2 = plot_best_result(ax2, savedir + 'SI Rule Split/', SI_TDPS, col=[1,0,1], label='Split SIRule')
    b4 = plot_best_result(ax2, savedir + 'EWC Rule Split/', EWC_TDPS, col=[0,1,0],label='Split EWC+Rule')
    b5 = plot_best_result(ax2, savedir + 'SI XdG/', SI_TD, col=[1,0,1], label='SI+XdG',linestyle='--')
    b6 = plot_best_result(ax2, savedir + 'EWC XdG/', EWC_TD, col=[0,1,0], label='EWC+XdG',linestyle='--')

    accuracy['SI_Rule_Split'] = b1
    accuracy['EWC_Rule_Split'] = b4

    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [ylim_min,1],[0,100],[])

    ax2.legend(ncol=3, fontsize=9)
    ax2.grid(True)
    ax2.set_xlim(0,100)
    add_subplot_details(ax2, [ylim_min,1],[0,100],[])

    plt.tight_layout()
    plt.savefig('Figure2_control.pdf', format='pdf')
    plt.show()


    """

    # Figure 2B
    # Plotting: SI+TD Partial, SI, EWC+TD Partial, EWC
    # No dropout on input layer



    b2 = plot_best_result(ax2, savedir + 'SI/', SI, col=[1,0,1], label='SI')
    b4 = plot_best_result(ax2, savedir + 'EWC Fixed/', EWC, col=[0,1,0], label='EWC')
    b1 = plot_best_result(ax2, savedir + 'SI Rule/', SI_TDP, col=[1,0,1], label='SI+Partial',linestyle='--')
    b3 = plot_best_result(ax2, savedir + 'EWC Rule/', EWC_TDP, col=[0,1,0], label='EWC+Partial',linestyle='--')

    accuracy['SI_Partial'] = b1
    accuracy['EWC_Partial'] = b3

    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(True)
    ax2.set_xlim(0,100)
    add_subplot_details(ax2, [ylim_min,1],[0,100],[])


    # Figure 2C
    # Plotting: SI+TD Partial Split, SI+TD Partial, EWC+TD Partial, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TDPS  = 'mnist_SI_split_rule'
    EWC_TDPS = 'split_InpDO_mnist_EWC'

    b1 = plot_best_result(ax3, savedir + 'SI Rule/', SI_TDP, col=[1,0,1], label='SI+Partial',linestyle='--')
    b3 = plot_best_result(ax3, savedir + 'EWC Rule/', EWC_TDP, col=[0,1,0], label='EWC+Partial',linestyle='--')
    b2 = plot_best_result(ax3, savedir + 'SI Rule Split/', SI_TDPS, col=[1,0,1], label='Split SI+Partial')
    b4 = plot_best_result(ax3, savedir + 'EWC Split/', EWC_TDPS, col=[0,1,0],label='Split EWC+Partial')

    accuracy['SI_Split'] = b2
    accuracy['EWC_Split'] = b4

    ax3.legend(ncol=2, fontsize=9)
    ax3.grid(True)
    ax3.set_xlim(0,100)
    add_subplot_details(ax3, [ylim_min,1],[0,100],[])


    # Figure 2D
    # Plotting: SI+TD Partial Split, SI+TD Full, EWC+TD Full, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TD    = 'neurons_InpDO_mnist_SI'
    EWC_TD   = 'neurons_InpDO_mnist_EWC'

    b2 = plot_best_result(ax4, savedir + 'SI Rule Split/', SI_TDPS, col=[1,0,1], label='Split SI++Partial')
    b4 = plot_best_result(ax4, savedir + 'EWC Split/', EWC_TDPS, col=[0,1,0],label='Split EWC++Partial')
    b1 = plot_best_result(ax4, savedir + 'SI XdG/', SI_TD, col=[1,0,1], label='SI+XdG',linestyle='--')
    b3 = plot_best_result(ax4, savedir + 'EWC XdG/', EWC_TD, col=[0,1,0], label='EWC+XdG',linestyle='--')

    accuracy['SI_XdG'] = b1
    accuracy['EWC_XdG'] = b3

    ax4.grid(True)
    ax4.set_xlim(0,100)
    add_subplot_details(ax4, [ylim_min,1],[0,100],[])
    ax4.legend(ncol=2, fontsize=9)

    """



    for k, v in accuracy.items():
        print(k, ' = ', v)



def plot_RNN_fig():

    savedir = '/home/masse/Short-term-plasticity-RNN/savedir/CL3/'
    all_same_scale = True

    f, ax1 = plt.subplots(1,1,figsize=(4,3))

    # Figure 2A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    SI      = 'RNN_CL_SI_short_h500_gating0'
    SI_TD    = 'RNN_CL_SI_short_h500_gating75'


    b0 = plot_best_result(ax1, savedir, SI, col=[0,0,1], label='SI')
    b1 = plot_best_result(ax1, savedir, SI_TD, col=[0,1,0], label='SI+TD Full')

    print('A: ', b0 , b1)


    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,17)
    add_subplot_details(ax1, [0.5,1],[0,25],[])
    ax1.set_xticks([4,8,12,16,20,24])


    plt.tight_layout()
    plt.savefig('Figure_ONR.pdf', format='pdf')
    plt.show()

def plot_ONR_fig1():

    savedir = './savedir/mnist/'
    all_same_scale = True

    f, (ax1,ax2) = plt.subplots(2,1,figsize=(3,6))

    # Figure 2A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    #base    = 'mnist_no_stabilization'
    EWC     = 'mnist_EWC'
    SI      = 'mnist_SI'
    SI_TD    = 'neurons_InpDO_mnist_SI'
    EWC_TD   = 'neurons_InpDO_mnist_EWC'


    b3 = plot_best_result(ax1, savedir, SI, col=[0,0,1], label='SI')
    b2 = plot_best_result(ax1, savedir, EWC, col=[1,0,0], label='EWC')
    b5 = plot_best_result(ax1, savedir, EWC_TD, col=[1,0,1], label='EWC+TD Full',linestyle='--')
    b4 = plot_best_result(ax1, savedir, SI_TD, col=[0,1,0], label='SI+TD Full')

    print('A: ', b2, b3, b4, b5)


    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [0.0,1],[0,100],[])

    savedir = './savedir/cifar/'

    #base    = 'cifar_no_stabilization'
    EWC     = 'cifar_EWC'
    SI      = 'cifar_SI'
    SI_TD    = 'neurons_cifar_SI'
    EWC_TD   = 'neurons_cifar_EWC'

    #b1 = plot_best_result(ax2, savedir, base, col=[0,0,0], label='base')
    b2 = plot_best_result(ax2, savedir, EWC, col=[1,0,0], label='EWC')
    b3 = plot_best_result(ax2, savedir, SI, col=[0,0,1], label='SI')
    b4 = plot_best_result(ax2, savedir, SI_TD, col=[0,1,0], label='SI+TD Full')
    b5 = plot_best_result(ax2, savedir, EWC_TD, col=[1,0,1], label='EWC+TD Full',linestyle='--')

    print('B: ', b2, b3, b4, b5)

    ax2.legend(ncol=3, fontsize=9)
    ax2.grid(True)
    add_subplot_details(ax2, ylim = [0,1], xlim = [0,20])


    plt.tight_layout()
    plt.savefig('Figure_ONR.pdf', format='pdf')
    plt.show()

def plot_fig3():

    savedir = './savedir/mnist/'
    all_same_scale = True
    f, ax1 = plt.subplots(1,1,figsize=(4,3))
    accuracy = {}

    # Figure 2A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    SI_5  = 'neurons_InpDO_mnist_SI'
    SI_2  = 'neurons_InpDO_mnist_SI_1of2'
    SI_3  = 'neurons_InpDO_mnist_SI_1of3'
    SI_4  = 'neurons_InpDO_mnist_SI_1of4'
    SI_6  = 'neurons_InpDO_mnist_SI_1of6'
    #SI= 'neurons_InpDO_mnist_SI'

    b1 = plot_best_result(ax1, savedir, SI_2, col=[1,0,0], label='SI_2')
    b2 = plot_best_result(ax1, savedir, SI_3, col=[0,0,1], label='SI_3')
    b3 = plot_best_result(ax1, savedir, SI_4, col=[0,1,0], label='SI_4')
    b4 = plot_best_result(ax1, savedir, SI_5, col=[0,0,0], label='SI_5')
    b5 = plot_best_result(ax1, savedir, SI_6, col=[1,0,1], label='SI_6')


    accuracy['SI_2'] = b1
    accuracy['SI_3'] = b2
    accuracy['SI_4'] = b3
    accuracy['SI_5'] = b4
    accuracy['SI_6'] = b5


    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,500)
    add_subplot_details(ax1, [0.9,1],[0,100],[])

    plt.tight_layout()
    plt.savefig('Figure3.pdf', format='pdf')
    plt.show()

    for k, v in accuracy.items():
        print(k , ' = ', v)

def plot_fig4():

    savedir = '/home/masse/Spin-TD-Network/savedir/mnist/500perms/'
    all_same_scale = True
    f, ax1 = plt.subplots(1,1,figsize=(4,3))
    accuracy = {}

    # Figure 2A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    SI      = 'mnist_SI_500perms'
    SI_XCG  = 'neurons_InpDO_mnist_SI_500perms'
    #SI= 'neurons_InpDO_mnist_SI'

    b1 = plot_best_result(ax1, savedir, SI, col=[0,0,1], label='SI')
    b2 = plot_best_result(ax1, savedir, SI_XCG, col=[1,0,0], label='SI_XCG')

    accuracy['SI'] = b1
    accuracy['SI XdG'] = b2


    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,500)
    add_subplot_details(ax1, [0.5,1],[0,500],[])

    plt.tight_layout()
    plt.savefig('Figure4_v3.pdf', format='pdf')
    plt.show()

    for k, v in accuracy.items():
        print(k , ' = ', v)

def plot_fig2():

    #savedir = './savedir/mnist/'
    savedir = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/'
    savedir1 = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final MNIST/'
    all_same_scale = True
    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,6))
    accuracy = {}

    ylim_min = 0.6

    # Figure 2A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    base    = 'mnist_no_stabilization'
    EWC     = 'mnist_EWC'
    SI      = 'mnist_SI'
    #SI= 'neurons_InpDO_mnist_SI'

    b3 = plot_best_result(ax1, savedir + 'SI/', SI, col=[1,0,1], label='SI')
    b1 = plot_best_result(ax1, savedir + 'Baseline/', base, col=[0,0,0], label='Base')
    b2 = plot_best_result(ax1, savedir + 'EWC Fixed/', EWC, col=[0,1,0], label='EWC')

    accuracy['base'] = b1
    accuracy['EWC'] = b2
    accuracy['SI'] = b3

    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0,100)
    add_subplot_details(ax1, [ylim_min,1],[0,100],[])

    # Figure 2B
    # Plotting: SI+TD Partial, SI, EWC+TD Partial, EWC
    # No dropout on input layer
    SI_TDP  = 'mnist_SI_rule'
    EWC_TDP = 'mnist_EWC_rule'


    b2 = plot_best_result(ax2, savedir + 'SI/', SI, col=[1,0,1], label='SI')
    b4 = plot_best_result(ax2, savedir + 'EWC Fixed/', EWC, col=[0,1,0], label='EWC')
    b1 = plot_best_result(ax2, savedir + 'SI Rule/', SI_TDP, col=[1,0,1], label='SI+Partial',linestyle='--')
    b3 = plot_best_result(ax2, savedir + 'EWC Rule/', EWC_TDP, col=[0,1,0], label='EWC+Partial',linestyle='--')

    accuracy['SI_Partial'] = b1
    accuracy['EWC_Partial'] = b3

    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(True)
    ax2.set_xlim(0,100)
    add_subplot_details(ax2, [ylim_min,1],[0,100],[])


    # Figure 2C
    # Plotting: SI+TD Partial Split, SI+TD Partial, EWC+TD Partial, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TDPS  = 'mnist_SI_split_rule'
    EWC_TDPS = 'split_InpDO_mnist_EWC'

    b1 = plot_best_result(ax3, savedir + 'SI Rule/', SI_TDP, col=[1,0,1], label='SI+Partial',linestyle='--')
    b3 = plot_best_result(ax3, savedir + 'EWC Rule/', EWC_TDP, col=[0,1,0], label='EWC+Partial',linestyle='--')
    b2 = plot_best_result(ax3, savedir + 'SI Rule Split/', SI_TDPS, col=[1,0,1], label='Split SI+Partial')
    b4 = plot_best_result(ax3, savedir + 'EWC Split/', EWC_TDPS, col=[0,1,0],label='Split EWC+Partial')

    accuracy['SI_Split'] = b2
    accuracy['EWC_Split'] = b4

    ax3.legend(ncol=2, fontsize=9)
    ax3.grid(True)
    ax3.set_xlim(0,100)
    add_subplot_details(ax3, [ylim_min,1],[0,100],[])


    # Figure 2D
    # Plotting: SI+TD Partial Split, SI+TD Full, EWC+TD Full, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TD    = 'neurons_InpDO_mnist_SI'
    EWC_TD   = 'neurons_InpDO_mnist_EWC'
    XdG = 'mnist_nostab'

    #b2 = plot_best_result(ax4, savedir + 'SI Rule Split/', SI_TDPS, col=[1,0,1], label='Split SI++Partial')
    #b4 = plot_best_result(ax4, savedir + 'EWC Split/', EWC_TDPS, col=[0,1,0],label='Split EWC++Partial')
    b3 = plot_best_result(ax4, savedir + 'SI/', SI, col=[1,0,1], label='SI')
    b2 = plot_best_result(ax4, savedir + 'EWC Fixed/', EWC, col=[0,1,0], label='EWC')

    b1 = plot_best_result(ax4, savedir + 'SI XdG/', SI_TD, col=[1,0,1], label='SI+XdG',linestyle='--')
    b3 = plot_best_result(ax4, savedir + 'EWC XdG/', EWC_TD, col=[0,1,0], label='EWC+XdG',linestyle='--')
    b5 = plot_best_result(ax4, savedir + 'XdG/', XdG, col=[0,0,0], label='XdG',linestyle='--')

    accuracy['SI_XdG'] = b1
    accuracy['EWC_XdG'] = b3
    accuracy['XdG'] = b5

    ax4.grid(True)
    ax4.set_xlim(0,100)
    add_subplot_details(ax4, [ylim_min,1],[0,100],[])
    ax4.legend(ncol=2, fontsize=9)


    plt.tight_layout()
    plt.savefig('Figure2.pdf', format='pdf')
    plt.show()

    for k, v in accuracy.items():
        print(k, ' = ', v)


def mnist_table():
    savedir = './savedir/perm_mnist/archive'

    base     = 'mnist_n2000_no_stabilization'     # archive1
    SI       = 'perm_mnist_n2000_d1_no_topdown'   # archive0
    EWC      = 'mnist_n2000_EWC'                  # archive1

    SI_TDP   = 'perm_mnist_n2000_d1_bias'         # archive0
    EWC_TDP  = 'perm_mnist_n2000_d1_EWC_bias'     # archive1

    SI_TDPS  = 'mnist_n2000_pathint_split_oc'     # archive2
    EWC_TDPS = 'mnist_n2000_EWC_split_oc'         # archive2

    SI_TDF   = 'perm_mnist_n2000_d1_1of5'         # archive0
    EWC_TDF  = 'perm_mnist_n2000_d1_EWC_1of5'     # archive1

    archs = [1,0,1,0,1,2,2,0,1]
    names = ['Base', 'SI', 'EWC', 'SI + Partial', 'EWC + Partial', \
             'SI + Partial + Split', 'EWC + Partial + Split',\
             'SI + Full', 'EWC + Full']
    locs  = [base, SI, EWC, SI_TDP, EWC_TDP, SI_TDPS, EWC_TDPS, SI_TDF, EWC_TDF]

    with open('mnist_table_data.tsv', 'w') as f:
        f.write('Name\tC\tT1\tT10\tT20\tT50\tT100\n')
        for a, s, n in zip(archs, locs, names):
            c_opt, acc = retrieve_best_result(savedir+str(a)+'/', s)
            f.write(n + '\t' + str(c_opt) + '\t' + str(acc[0])
                                          + '\t' + str(acc[9])
                                          + '\t' + str(acc[19])
                                          + '\t' + str(acc[49])
                                          + '\t' + str(acc[99]) + '\n')


def cifar_table():
    savedir = './savedir/cifar_no_multihead/'

    base     = 'cifar_n1000_no_stabilization'
    SI       = 'cifar_n1000_pathint'
    EWC      = 'cifar_n1000_EWC'

    SI_TDP   = 'cifar_n1000_partial_pathint'
    EWC_TDP  = 'cifar_n1000_partial_EWC'

    SI_TDPS  = 'cifar_n1164_split_pathint'
    EWC_TDPS = 'cifar_n1164_split_EWC'

    SI_TDF   = 'cifar_n1000_full_pathint'
    EWC_TDF  = 'cifar_n1000_full_EWC'

    names = ['Base', 'SI', 'EWC', 'SI + Partial', 'EWC + Partial', \
             'SI + Partial + Split', 'EWC + Partial + Split',\
             'SI + Full', 'EWC + Full']
    locs  = [base, SI, EWC, SI_TDP, EWC_TDP, SI_TDPS, EWC_TDPS, SI_TDF, EWC_TDF]

    with open('cifar_table_data.tsv', 'w') as f:
        f.write('Name\tC\tT1\tT10\tT20\n')
        for s, n in zip(locs, names):
            c_opt, acc = retrieve_best_result(savedir, s)
            f.write(n + '\t' + str(c_opt) + '\t' + str(acc[0])
                                          + '\t' + str(acc[9])
                                          + '\t' + str(acc[19]) + '\n')



def fig2_inset():
    f, ax = plt.subplots(1,1)
    # Figure 2D
    # Plotting: SI+TD Partial Split, SI+TD Full, EWC+TD Full, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TD    = 'perm_mnist_n2000_d1_1of5'
    SI_TDPS  = 'mnist_n2000_pathint_split_oc'
    EWC_TD   = 'perm_mnist_n2000_d1_EWC_1of5'
    EWC_TDPS = 'mnist_n2000_EWC_split_oc'

    b2 = plot_best_result(ax, './savedir/perm_mnist/archive2/', SI_TDPS, col=[0.7,0.7,0], label='Split SI+TD Par.')
    b4 = plot_best_result(ax, './savedir/perm_mnist/archive2/', EWC_TDPS, col=[0.7,0.7,0.7],label='Split EWC+TD Par.')
    b1 = plot_best_result(ax, './savedir/perm_mnist/archive0/', SI_TD, col=[0,0,1], label='SI+TD Full')
    b3 = plot_best_result(ax, './savedir/perm_mnist/archive1/', EWC_TD, col=[0,1,0], label='EWC+TD Full')
    ax.grid(True)
    ax.set_yticks([0.85,0.90,0.95,1.0])
    ax.set_xlim(0,100)
    add_subplot_details(ax, [0.85,1],[])

    plt.tight_layout()
    plt.show()


def plot_fig5():

    savedir = '/home/masse/Context-Dependent-Gating/savedir/ImageNet/'
    savedir1 = '/media/masse/MySSDataStor1/Context-Dependent Gating/Final ImageNet/'
    f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,6))
    accuracy = {}

    # Figure 5A
    # Plotting: SI, EWC, base
    # No dropout on input layer
    base    = 'imagenet_base_omega'
    EWC     = 'ImageNet_EWC_omega'
    SI      = 'imagenet_SI_omega'

    base_MH = 'imagenet_baseMH_'
    EWC_MH  = 'ImageNet_EWC_MH_omega'
    SI_MH   = 'imagenet_SI_MH_'

    b1 = plot_best_result(ax1, savedir, base, col=[0,0,0], label='base')
    b2 = plot_best_result(ax1, savedir1+'EWC/', EWC, col=[0,1,0], label='EWC')
    b3 = plot_best_result(ax1, savedir, SI, col=[1,0,1], label='SI')

    b4 = plot_best_result(ax1, savedir, base_MH, col=[0,0,0], label='base MH', linestyle='--')
    b5 = plot_best_result(ax1, savedir1+'EWC MH/', EWC_MH, col=[0,1,0], label='EWC MH', linestyle='--')
    b6 = plot_best_result(ax1, savedir, SI_MH, col=[1,0,1], label='SI MH', linestyle='--')

    accuracy['base'] = b1
    accuracy['EWC'] = b2
    accuracy['SI'] = b3
    accuracy['base_MH'] = b4
    accuracy['EWC_MH'] = b5
    accuracy['SI_MH'] = b6

    print('A ', b1,b2,b3,b4,b5,b6)

    ax1.legend(ncol=3, fontsize=9)
    ax1.grid(True)
    add_subplot_details(ax1, ylim = [0,1], xlim = [0,100])

    # Figure 3B2
    # Plotting: SI+TD Partial, SI, EWC+TD Partial, EWC
    # No dropout on input layer
    SI_TDP  = 'ImageNet_SI_rule_'
    EWC_TDP = 'ImageNet_EWC_rule_' # InpDO was a typo, not actually using drop out on iputs

    b2 = plot_best_result(ax2, savedir, SI, col=[1,0,1], label='SI')
    b4 = plot_best_result(ax2, savedir1+'EWC/', EWC, col=[0,1,0], label='EWC')
    b1 = plot_best_result(ax2, savedir1+'SI Rule/', SI_TDP, col=[1,0,1], label='SI+Partial', linestyle='--')
    b3 = plot_best_result(ax2, savedir1+'EWC Rule/', EWC_TDP, col=[0,1,0], label='EWC+Partal', linestyle='--')

    accuracy['SI Partial'] = b1
    accuracy['EWC Partial'] = b3

    ax2.set_xlim(0,100)
    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(True)
    add_subplot_details(ax2, ylim = [0,1], xlim = [0,100])

    # Figure 3C
    # Plotting: SI+TD Partial Split, SI+TD Partial, EWC+TD Partial, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TDPS  = 'ImageNet_SI_rule_split_'
    EWC_TDPS = 'ImageNet_EWC_rule_split_'

    b1 = plot_best_result(ax3, savedir1+'SI Rule Split/', SI_TDP, col=[1,0,1], label='Split SI+TD Partial')
    b3 = plot_best_result(ax3, savedir1+'EWC Rule/', EWC_TDP, col=[0,1,0], label='EWC+Partal', linestyle='--')
    b2 = plot_best_result(ax3, savedir1+'SI Rule/', SI_TDP, col=[1,0,1], label='SI+Partial', linestyle='--')
    b4 = plot_best_result(ax3, savedir1+'EWC Rule Split/', EWC_TDPS, col=[0,1,0],label='Split EWC+Partial')

    accuracy['SI Split'] = b1
    accuracy['EWC Split'] = b4

    ax3.set_xlim(0,100)
    ax3.legend(ncol=2, fontsize=9)
    ax3.grid(True)
    add_subplot_details(ax3, ylim = [0,1], xlim = [0,100])

    # Figure 3D
    # Plotting: SI+TD Partial Split, SI+TD Full, EWC+TD Full, EWC+TD Partial Split
    # Dropout irrelevant?
    SI_TD    = 'ImageNet_SI_XdG_'
    EWC_TD   = 'ImageNet_EWC_XdG_'
    XdG = 'imagenet_nostab'

    #b2 = plot_best_result(ax4, savedir1+'SI Rule Split/', SI_TDP, col=[1,0,1], label='Split SI+TD Partial')
    #b4 = plot_best_result(ax4, savedir1+'EWC Rule Split/', EWC_TDPS, col=[0,1,0],label='Split EWC+Partial')

    b2 = plot_best_result(ax4, savedir1+'EWC/', EWC, col=[0,1,0], label='EWC')
    b3 = plot_best_result(ax4, savedir, SI, col=[1,0,1], label='SI')

    b1 = plot_best_result(ax4, savedir1+'EWC XdG/', EWC_TD, col=[0,1,0],label='EWC+XdG', linestyle='--')
    b3 = plot_best_result(ax4, savedir1+'SI XdG/', SI_TD, col=[1,0,1],label='SI+XdG', linestyle='--')

    b5 = plot_best_result(ax4, savedir1+'XdG/', XdG, col=[0,0,0],label='SI+XdG', linestyle='--')

    accuracy['SI XdG'] = b3
    accuracy['EWC XdG'] = b1
    accuracy['XdG'] = b5

    ax4.set_xlim(0,100)
    ax4.legend(ncol=2, fontsize=9)
    ax4.grid(True)
    add_subplot_details(ax4, ylim = [0,1], xlim = [0,100])

    plt.tight_layout()
    plt.savefig('Figure3.pdf', format='pdf')
    plt.show()

    for k, v in accuracy.items():
        print(k , ' = ', v)

def plot_fig5B():

    savedir = '/home/masse/Context-Dependent-Gating/savedir/ImageNet/'
    f, ax4 = plt.subplots(1,1,figsize=(8,6))
    accuracy = {}

    SI_TDPS  = 'imagenet_SI_split_'
    EWC_TDPS = 'imagenet_EWC_split_'
    SI_TD    = 'imagenet_SI_XdG_'
    EWC_TD   = 'imagenet_EWC_XdG_'

    b2 = plot_best_result(ax4, savedir, SI_TDPS, col=[1,0,1], label='Split SI+Partial')
    b4 = plot_best_result(ax4, savedir, EWC_TDPS, col=[0,1,0],label='Split EWC+Partial')
    b1 = plot_best_result(ax4, savedir, EWC_TD, col=[0,1,0],label='EWC+XdG', linestyle='--')
    b3 = plot_best_result(ax4, savedir, SI_TD, col=[1,0,1],label='SI+XdG', linestyle='--')

    accuracy['SI XdG'] = b1
    accuracy['EWC XdG'] = b3

    ax4.set_xlim(0,100)
    ax4.legend(ncol=2, fontsize=9)
    ax4.grid(True)
    add_subplot_details(ax4, ylim = [0,1], xlim = [0,100])

    plt.tight_layout()
    plt.savefig('Figure3.pdf', format='pdf')
    plt.show()

def plot_ARL_fig():

    savedir = '/home/masse/Context-Dependent-Gating/savedir/ImageNet/'
    f, ax4 = plt.subplots(1,1,figsize=(8,6))
    accuracy = {}

    SI  = 'imagenet_SI_omega'
    SI_TD    = 'imagenet_SI_XdG_'


    b1 = plot_best_result(ax4, savedir, SI, col=[0,0,1], label='SI')
    b2 = plot_best_result(ax4, savedir, SI_TD, col=[1,0,0],label='SI+XdG')

    accuracy['SI'] = b1
    accuracy['SI_XdG'] = b2

    ax4.set_xlim(0,100)
    ax4.legend(ncol=2, fontsize=9)
    ax4.grid(True)
    add_subplot_details(ax4, ylim = [0,0.7], xlim = [0,100])

    plt.tight_layout()
    plt.savefig('Figure_ARL.pdf', format='pdf')
    plt.show()

def plot_mnist_figure():

    f = plt.figure(figsize=(6,2.5))

    # SI only, no top-down
    SI_fn = 'perm_mnist_n2000_d1_no_topdown_omega'
    # SI only + top-down
    SI_td_fn = 'perm_mnist_n2000_d1_bias_omega'
    # SI only + split in 5 + top-down
    SI_split_fn = 'perm_mnist_n735_d1_bias_20tasks'
    # SI + INH TD, selecting one out of 4
    SI_inh_td4_fn = 'perm_mnist_n2000_d1_1of4_omega'
    # SI + INH TD, selecting one out of 5
    SI_inh_td5_fn = 'perm_mnist_n2000_d1_1of5_omega'
    # SI + INH TD, selecting one out of 6
    SI_inh_td6_fn = 'perm_mnist_n2000_d1_1of6_omega'
    # SI + INH TD, selecting one out of 3
    SI_inh_td3_fn = 'perm_mnist_n2000_d1_1of3_omega'
    # SI + INH TD, selecting one out of 2
    SI_inh_td2_fn = 'perm_mnist_n2000_d1_1of2_omega'

    ax = f.add_subplot(1, 2, 1)
    plot_best_result(ax, data_dir, SI_fn, col = [0,0,1], description = 'SI only')
    plot_best_result(ax, data_dir, SI_td_fn, col = [1,0,0])
    plot_best_result(ax, data_dir, SI_split_fn, col = [0,1,0], split = 5)
    plot_best_result(ax, data_dir, SI_inh_td5_fn, col = [1,0,1], description = 'SI + TD date 80%')
    add_subplot_details(ax, [0.8, 1],[0.85, 0.9,0.95])

    ax = f.add_subplot(1, 2, 2)
    plot_best_result(ax, data_dir, SI_inh_td5_fn, col = [1,0,1])
    plot_best_result(ax, data_dir, SI_inh_td4_fn, col = [0,1,0])
    plot_best_result(ax, data_dir, SI_inh_td6_fn, col = [0,0,1], description = 'SI + TD date 86.67%')
    plot_best_result(ax, data_dir, SI_inh_td3_fn, col = [0,1,1])
    plot_best_result(ax, data_dir, SI_inh_td2_fn, col = [0,0,0])
    add_subplot_details(ax, [0.9, 1], [0.95])

    plt.tight_layout()
    plt.savefig('Fig1.pdf', format='pdf')
    plt.show()

def add_subplot_details(ax, ylim = [0,1], xlim = [0,100],yminor = []):

    d = ylim[1] - ylim[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_ylim([ylim[0], ylim[1]])
    for i in yminor:
        ax.plot([0,100],[i,i],'k--')
    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylabel('Mean task accuracy')
    ax.set_xlabel('Task number')

def plot_best_result(ax, data_dir, prefix, col = [0,0,1], split = 1, description = [], label=None, linestyle = '-'):


    # Get filenames
    name_and_data = []
    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix):
            x = pickle.load(open(data_dir + full_fn, 'rb'))
            name_and_data.append((full_fn, x['accuracy_full'][-1], x['par']['omega_c']))

    if prefix == 'mnist_SI_rule':
        print('mnist_STI_rule')
        print(name_and_data)
        print(os.listdir(data_dir))
        print(data_dir)
        print('XXXX')

    # Find number of c's and v's
    cids = []
    vids = []
    xids = []
    om_c = []
    for (f, _, oc) in name_and_data:

        if 'xi' in f:
            if f[-12] not in cids:
                cids.append(f[-12])
            if f[-8] not in xids:
                xids.append(f[-8])
            if f[-5] not in vids:
                vids.append(f[-5])
        else:
            if f[-9].isdigit():
                c = f[-9:-7]
            else:
                c = f[-8]
            if c not in cids:
                cids.append(c)
            if f[-5] not in vids:
                vids.append(f[-5])
            xids = [0]
        om_c.append(oc)


    accuracies = np.zeros((len(xids),len(cids)))
    count = np.zeros((len(xids),len(cids)))
    cids = sorted(cids)
    vids = sorted(vids)
    xids = sorted(xids)

    print(prefix, cids, vids, xids)

    for i, c_id, in enumerate(cids):
        for v_id in vids:
            for j, x_id in enumerate(xids):
                #text_c = 'omega'+str(c_id)
                text_c = 'omega'+str(c_id)
                text_v = '_v'+str(v_id)
                text_x = '_xi'+str(x_id)
                for full_fn in os.listdir(data_dir):
                    if full_fn.startswith(prefix) and 'xi' in full_fn and text_c in full_fn and text_v in full_fn and text_x in full_fn:
                        #print('c_id', c_id)
                        x = pickle.load(open(data_dir + full_fn, 'rb'))
                        accuracies[j,i] += x['accuracy_full'][-1]
                        count[j,i] += 1
                    elif full_fn.startswith(prefix) and not 'xi' in full_fn  and text_c in full_fn and text_v in full_fn:
                        #print('c_id', c_id)
                        x = pickle.load(open(data_dir + full_fn, 'rb'))
                        accuracies[j,i] += x['accuracy_full'][-1]
                        count[j,i] += 1

    accuracies /= (1e-16+count)
    accuracies = np.reshape(accuracies,(1,-1))
    print(prefix)
    print(accuracies)
    ind_best = np.argsort(accuracies)[-1]
    best_c = int(ind_best[-1]%len(cids))
    best_xi = ind_best[-1]//len(cids)
    task_accuracy = []


    for v_id in vids:
        #text_c = 'omega'+str(cids[best_c])
        text_c = 'omega'+str(cids[best_c])
        text_xi = 'xi'+str(xids[best_xi])
        text_v = '_v'+str(v_id)

        print(prefix, text_c, text_xi, text_v)

        for full_fn in os.listdir(data_dir):
            if full_fn.startswith(prefix)  and 'xi' in full_fn and text_c in full_fn and text_v in full_fn and text_x in full_fn:
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                task_accuracy.append(x['accuracy_full'])
                print(prefix,' ', full_fn, ' ', x['par']['stabilization'], ' omega C ', x['par']['omega_c'])
            elif full_fn.startswith(prefix)  and not 'xi' in full_fn and text_c in full_fn and text_v in full_fn:
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                task_accuracy.append(x['accuracy_full'])
                print(prefix,' ', full_fn, ' ', x['par']['stabilization'], x['par']['gating_type'], x['par']['multihead'], ' omega C ', x['par']['omega_c'], x['par']['omega_xi'])

    task_accuracy = np.mean(np.stack(task_accuracy),axis=0)


    if split > 1:
        task_accuracy = np.array(task_accuracy)
        task_accuracy = np.tile(np.reshape(task_accuracy,(-1,1)),(1,split))
        task_accuracy = np.reshape(task_accuracy,(1,-1))[0,:]

    if not description == []:
        print(description , ' ACC after 10 trials = ', task_accuracy[9],  ' after 30 trials = ', task_accuracy[29],  \
            ' after 100 trials = ', task_accuracy[99])

    ax.plot(np.arange(1, np.shape(task_accuracy)[0]+1), task_accuracy, color = col, linestyle = linestyle, label=label)


    return task_accuracy[[9,-1]]

def retrieve_best_result(data_dir, fn):
    best_accuracy = -1
    val_c = 0.
    for f in os.listdir(data_dir):
        if f.startswith(fn):
            x = pickle.load(open(data_dir+f, 'rb'))
            if x['accuracy_full'][-1] > best_accuracy:
                best_accuracy = x['accuracy_full'][-1]
                task_accuracy = x['accuracy_full']
                val_c         = x['par']['omega_c']

    return val_c, task_accuracy
