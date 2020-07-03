import json

import click

import GA_optimizers.ECGA as ecga
import GA_optimizers.PSO as pso
import GA_optimizers.sGA as sga
from arch_eval import evaluate_architecture
from data_dict import load_data, datasets_dict
from genetic_CNN import code_length_of

opt_dict = {'pso': pso, 'ecga': ecga, 'sga': sga}


def hyper_parameters(optimizer, epochs, num_filters, fc_nodes,
                     drop_out, kernel, stride, pooling, pool_size):
    hyper_params = {}
    hyper_params['optimizer'] = optimizer
    hyper_params['epochs'] = epochs
    hyper_params['kernel size'] = (kernel, kernel)
    hyper_params['strides'] = (stride, stride)
    hyper_params['drop out'] = drop_out
    hyper_params['pooling'] = pooling
    hyper_params['filters'] = num_filters
    hyper_params['fc units'] = fc_nodes
    hyper_params['pool size'] = (pool_size, pool_size)

    return hyper_params


class Config(object):
    def __init__(self):
        self.dict = {}


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option('--nodes', '-n', required=True, help='Number of nodes k for each stages S. Ex: [4,5]')
@click.option('--dataset', '-dtset', required=True, type=click.Choice(datasets_dict.keys(), case_sensitive=False))
@click.option('--network_optimizer', '-nn_opt', default='adam', type=str)
@click.option('--epochs', '-ep', default=10, type=int)
@click.option('--num_filters', '-numf', default=8, type=int)
@click.option('--fc_nodes', '-fc', default=128, type=int)
@click.option('--kernel_size', '-ksize', default=5, type=int)
@click.option('--pool_size', '-psize', default=2, type=int)
@click.option('--stride', '-s', default=2, type=int)
@click.option('--drop_out', '-do', default=0.5, type=click.FloatRange(0, 1))
@click.option('--pooling', '-pool', default='max', type=click.Choice(['max', 'average', 'avg']))
@pass_config
def cnn_model(config, nodes, dataset, network_optimizer, epochs, pool_size,
              kernel_size, stride, drop_out, pooling, num_filters, fc_nodes):
    """ CNN model Config """
    nodes = json.loads(nodes)
    nodes = [int(node) for node in nodes]
    hyper_params = hyper_parameters(network_optimizer, epochs, num_filters, fc_nodes,
                                    drop_out, kernel_size, stride, pooling, pool_size)
    data = load_data(dataset, image=True, small_data=False)
    config.dict['nodes'] = nodes
    config.dict['data'] = data
    config.dict['hyper_params'] = hyper_params


@cnn_model.command()
@click.option('--optimizer', '-opt', required=True, type=click.Choice(opt_dict.keys(), case_sensitive=False),
              help='Choose optimization method')
@click.option('--seed', '-s', default=1, type=int,
              help='Random seed for the random number generator (default value : 1)')
@click.option('--gen', '-g', default=-1, type=int,
              help='Max generations to evaluate population (default value : -1)')
@click.option('--num_inds', '-N', default=20,
              help='Number of genomes in population (default 20)')
@click.option('--tsize', '-ts', default=4, type=int,
              help='Define tournament size for some method')
@click.option('--mode', '-m', type=str,
              help='Choose which mode in optimization method to use')
@click.option('--printscr', '-prnscr', default=True, type=bool,
              help='Print result to command line')
@click.option('--log', '-l', default=True, type=bool,
              help='Log result to csv file')
@pass_config
def GA_config(config, optimizer, seed, gen, num_inds,
              log, tsize, mode, printscr):
    """ Genetic Algorithm Configuration """
    f_func = evaluate_architecture(config.dict['nodes'], config.dict['hyper_params'],
                                   config.dict['data'])
    f_dict = {'name': 'architecture evaluate',
              'd': sum(list(map(code_length_of, config.dict['nodes']))),
              'D': (0, 1),
              'real valued': False,
              'multi dims': True,
              'global maximum': None,
              'global minimum': None,
              'function': f_func}
    params = opt_dict[optimizer].get_parameters(N=num_inds, s=seed, g=gen,
                                                mode=mode, f=f_dict,
                                                maximize=True, t_size=tsize)
    result = opt_dict[optimizer].optimize(params, 0, printscr, log)
    print(result)
