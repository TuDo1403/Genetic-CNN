import numpy as np
import pandas as pd

from GA_optimizers.GA import *
from utils.plot import *

def variate(pop, crossover_mode):
    (num_inds, num_params) = pop.shape
    indices = np.arange(num_inds)

    offsprings = []
    np.random.shuffle(indices)

    for i in range(0, num_inds, 2):
        idx1, idx2 = indices[i], indices[i+1]
        offs1, offs2 = pop[idx1].copy(), pop[idx2].copy()

        if crossover_mode == 'onepoint':
            point = np.random.randint(low=0, high=num_params-1)
            offs1[:point], offs2[:point] = offs2[:point], offs1[:point].copy()
        else:
            for j in range(num_params):
                if np.random.randint(low=0, high=2) == 1:
                    offs1[j], offs2[j] = offs2[j], offs1[j]

        offsprings.append(offs1)
        offsprings.append(offs2)

    return np.reshape(offsprings, (num_inds, num_params))

def tournament_selection(f_pool, tournament_size, selection_size, maximize=False):
    num_inds = len(f_pool)
    indices = np.arange(num_inds)
    selected_indices = []
    comparer = max if maximize else min

    while len(selected_indices) < selection_size:
        np.random.shuffle(indices)

        for i in range(0, num_inds, tournament_size):
            idx_tournament = indices[i : i+tournament_size]
            best_idx = list(filter(lambda idx : f_pool[idx] == comparer(f_pool[idx_tournament]), idx_tournament))
            selected_indices.append(np.random.choice(best_idx))

    return selected_indices


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def optimize(params, plot=False, print_scr=True, log=False):
    """

    """

    # Initialize required parameters from dictionary
    num_inds = params['N']
    tournament_size = params['ts']
    max_gen = params['g']
    seed = params['s']
    maximize = params['maximize']
    crossover_mode = params['cm']

    f_dict = params['f']    # Dictionary of fitness function data
    real_valued = f_dict['real valued']
    (lower_bound, upper_bound) = f_dict['D']
    num_params = f_dict['d']

    # Plot search space
    plottable = plot and num_params == 2
    if plottable:
        data = get_plot_data(f_dict)
        fig, ax = plt.subplots()
        if plottable and plot == 2:
            ax = Axes3D(fig)

    # Write log file header
    if log:
        with open('../log/{}.txt'.format('sga'), 'w+') as f:
            f.write('{},{},{},{},{},{}\n'.format('gen', 'max result', 'min result', 'mean', 'std', 'best genome'))

    # Initialize
    comparer = np.argmax if maximize else np.argmin
    np.random.seed(seed)
    epsilon = 10**-5
    pop = initialize(num_inds, num_params, 
                     domain=[lower_bound, upper_bound], 
                     real_valued=real_valued)
    f_pop = evaluate(pop, f_dict['function'])
    selection_size = len(pop)
    gen = 0
    num_f_func_calls = len(f_pop)
    #
    while not pop_converge(pop):
        gen += 1
        if max_gen_reached(gen, max_gen):
            break

        # Variate
        offs = variate(pop, crossover_mode)

        # Evaluate
        f_offs = evaluate(offs, f_dict['function'])    
        num_f_func_calls += len(f_offs)

        # Selection
        pool = np.vstack((pop, offs))
        f_pool = np.hstack((f_pop, f_offs))

        pool_indices = tournament_selection(f_pool, tournament_size, 
                                            selection_size, maximize)
        pop = pool[pool_indices]
        f_pop = f_pool[pool_indices]
        #

        # Visualize / log result
<<<<<<< HEAD
        if print_scr or log:
            best_genome = pop[comparer(f_pop)]
            max_accuracy, min_accuracy = f_pop.max(), f_pop.min()
=======
        if print_scr and gen % 100 == 0:
            print('## Gen {}: {} (Fitness: {})'.format(gen, pop[comparer(f_pop)].reshape(1, -1), 
                                                       f_pop[comparer(f_pop)]))

        if log:
            best_genome = pop[comparer(f_pop)]
            max_result, min_result = f_pop.max(), f_pop.min()
>>>>>>> 363e80ec181653310f85dcbca99288c07200e9d6
            mean, std = f_pop.mean(), f_pop.std()
            if print_scr:
                print('Gen {}: best architecture : {} - accuracy (max/min) : {}/{} - mean/std : {}/{}'.format(gen, best_genome.reshape(1, -1)[0],
                                                                                                        max_accuracy, min_accuracy, mean, std))
            if log:
                with open('../log/{}.txt'.format('sga'), 'a+') as f:
                    f.write('{},{},{},{},{},{}\n'.format(gen, max_accuracy, min_accuracy, mean, std, best_genome))

        if plottable:
            ax.clear()
            if plot == 1:
                contour_plot(ax, data, f_dict, hold=True)
                ax_lim = (xlim, ylim) = ax.get_xlim(), ax.get_ylim()
                scatter_plot(ax_lim, ax, pop, hold=True)
            else:
                contour_3D(ax, data, f_dict, hold=True)
                ax_lim = (xlim, ylim, zlim) = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
                scatter_3D(ax_lim, ax, pop, f_pop, hold=True)
            plt.pause(epsilon)
        #

    if plottable: 
        plt.show()

    

    solution =  pop[comparer(f_pop)].reshape(1, -1).flatten()
    opt_sol_found = None

    optimize_goal = 'global maximum' if maximize else 'global minimum'
    if type(f_dict[optimize_goal]) != type(None):
        epsilon = 10**-5
        diffs = np.abs(f_dict[optimize_goal] - solution).sum(axis=1)
        opt_sol_found = len(np.where(diffs <= num_params*epsilon)[0]) != 0

    result = { 'solution' : solution, 
               'evaluate function calls' : num_f_func_calls, 
               'global optima found' : opt_sol_found }

    return result


def get_parameters(**params):
    mode = 'ux' if params['mode'] == "" or params['mode'] not in ['ux', 'onepoint'] else params['mode']
    default_params = { 'N' : params['N'],
                       's' : params['s'],
                       'g' : params['g'],
                       'ts' : params['t_size'],
                       'maximize' : params['maximize'],
                       'f' : params['f'],
                       'cm' : mode }
    return default_params