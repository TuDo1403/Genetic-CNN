import click

from numpy import zeros, ones
from data_dict import *

import PSO as pso
import ECGA as ecga
import sGA as sga

optimizers_dict = { 'pso' : pso, 'ecga' : ecga, 'sga' : sga }
data_dict = { 'cifar10' : cifar10_dict}

# @click.option('--maximize', '-max', default=False, type=bool, 
            # help='Define whether to maximize or minimize the output')
# @click.option('--function', '-f', required=True, type=str, 
            # help='Choose which function to evaluate')
# @click.option('--pop_shape', '-ps', default=(100, 2), type=(int, int), 
#             help='Define initial population shape (N, d) (default value : (100, 2))')
# @click.option('--plot', '-plt', default=0, type=click.IntRange(0, 3), 
            # help='0 (no plot), 1 (2d plot), 2 (3d plot)')
@click.command()
@click.option('--optimizer', '-opt', required=True, type=str, 
            help='Choose optimization method')
@click.option('--nodes', '-n', required=True, nargs='--states', type=int)
@click.option('--dataset', '-dt', required=True, type=str)
@click.option('--num_inds', '-N', default=20, type=click.IntRange(20, 100))
@click.option('--seed', '-s', default=1, type=int, 
            help='Random seed for the random number generator (default value : 1)')
@click.option('--gen', '-g', default=-1, type=int, 
            help='Max generations to evaluate population (default value : -1)')
@click.option('--tournament_size', '-ts', default=4, type=int, 
            help='Define tournament size for some method')
@click.option('--mode', type=str, 
            help='Choose which mode in optimization method to use')
@click.option('--print_scr', '-prnscr', default=True, type=bool, 
            help='Print result to command line')
@click.option('--network_optimizer', '-nn_opt', default='adam', type=str)
@click.option('--epochs', 'e', default=5, type=click.IntRange(1, 20))
@click.option('--batch_size', 'bs', default=64, type=int)
def run(optimizer, nodes, dataset, seed, gen, 
        tournament_size, mode, print_scr, num_inds):

    params = optimizers_dict[optimizer].get_parameters(N=num_inds, s=seed, g=gen,
                                                    mode=mode)
    pass

if __name__ == '__main__':
    run()