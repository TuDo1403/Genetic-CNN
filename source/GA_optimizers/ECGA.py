import numpy as np

from GA_optimizers.GA import *
from utils.plot import *

from numpy import arange, newaxis
def compute_minimum_description_length(pop, model):
    """ Compute minimum description length
        
    """

    # Compute model complexity
    N = len(pop)
    S = np.array(list(map(len, model)))
    MC = np.log2(N + 1) * np.sum(2**S - 1)

    # Compute compressed population complexity
    entropy = 0
    num_groups = len(model)
    events = [arange(2**S[i])[:, newaxis] >> np.arange(S[i])[::-1] & 1 for i in range(num_groups)]
    for i in range(num_groups):
        for event in events[i]:
            group = pop[:, model[i]]
            match = np.sum(group == event, axis=1)
            prob = np.count_nonzero(match == len(event)) / (N+1)
            if prob != 0:
                entropy += prob * np.log2(1/prob)
            
    CPC = N * entropy
    return CPC + MC


def generate_models_from(current_model):
    new_models = []
    for i in range(len(current_model) - 1):
        for j in range(i+1, len(current_model)):
            new_group = current_model.copy()
            del new_group[i]
            del new_group[j - 1]
            new_group.append(current_model[i] + current_model[j])
            new_models.append(new_group)

    return new_models

def compute_marginal_product_model(pop, model):
    current_MDL = compute_minimum_description_length(pop, model)
    new_models = generate_models_from(model)
    new_MDLs = np.array([compute_minimum_description_length(pop, model) for model in new_models])
    return model if current_MDL < np.min(new_MDLs) else new_models[np.argmin(new_MDLs)]


def variate(pop, model):
    (num_inds, num_params) = pop.shape
    indices = np.arange(num_inds)

    offs = []
    np.random.shuffle(indices)

    for i in range(0, num_inds, 2):
        idx1 = indices[i]
        idx2 = indices[i+1]
        offs1 = pop[idx1].copy()
        offs2 = pop[idx2].copy()

        for group in model:
            if np.random.rand() < 0.5:
                offs1[group], offs2[group] = offs2[group].copy(), offs1[group]
            
        offs.append(offs1)
        offs.append(offs2)

    return np.reshape(offs, (num_inds, num_params))

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
def optimize(params, plot=False, print_scr=False, log=False):
    """

    """

    # Initialize required parameters from dictionary
    num_inds = params['N']
    tournament_size = params['ts']
    max_gen = params['g']
    seed = params['s']
    maximize = params['maximize']

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
        with open('../log/{}.txt'.format('ecga'), 'w+') as f:
            f.write('{},{},{},{},{},{}\n'.format('gen', 'max result', 'min result', 'mean', 'std', 'best genome'))

    # Initialize
    comparer = np.argmax if maximize else np.argmin
    np.random.seed(seed)
    epsilon = 10**-5
    pop = initialize(num_inds, num_params, 
                     domain=[lower_bound, upper_bound], 
                     real_valued=real_valued)
    f_pop = evaluate(pop, f_dict['function'])
    pop_model = [[group] for group in np.arange(num_params)]
    selection_size = len(pop)
    gen = 0
    num_f_func_calls = len(f_pop)
    #
    while not pop_converge(pop):
        gen += 1
        if max_gen_reached(gen, max_gen):
            break

        # Model building
        while len(pop_model) != 1:
            model = compute_marginal_product_model(pop, pop_model)
            if model_converge(model, pop_model):
                break
            else:
                pop_model = model

        # Variate
        offs = variate(pop, pop_model)

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
        if print_scr or log:
            best_genome = pop[comparer(f_pop)]
            max_accuracy, min_accuracy = f_pop.max(), f_pop.min()
            mean, std = f_pop.mean(), f_pop.std()
            if print_scr:
                print('Gen {}: best architecture : {} - accuracy (max/min) : {}/{} - mean/std : {}/{}'.format(gen, best_genome.reshape(1, -1)[0],
                                                                                                        max_accuracy, min_accuracy, mean, std))
            if log:
                with open('../log/{}.txt'.format('ecga'), 'a+') as f:
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
        diffs = np.abs(f_dict[optimize_goal] - solution).sum(axis=1)
        opt_sol_found = len(np.where(diffs <= num_params*epsilon)[0]) != 0

    result = { 'solution' : solution, 
               'model' : pop_model,
               'evaluate function calls' : num_f_func_calls, 
               'global optima found' : opt_sol_found }
    return result

def model_converge(current_model, new_model):
    for group in current_model:
        if group not in new_model:
            return False
    return True

def get_parameters(**params):
    default_params = { 'N' : params['N'],
                       's' : params['s'],
                       'g' : params['g'],
                       'ts' : params['t_size'],
                       'maximize' : params['maximize'],
                       'f' : params['f']}
    return default_params