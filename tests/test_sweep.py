import time
import random
import argparse
import configparser
import ast

import numpy as np
import torch

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import pufferlib
import pufferlib.sweep
import pufferlib.utils


def synthetic_basic_task(args):
    train_args = args['train']
    learning_rate = train_args['learning_rate']
    total_timesteps = train_args['total_timesteps']
    score = np.exp(-(np.log10(learning_rate) + 3)**2)
    cost = total_timesteps / 50_000_000
    return score, cost

def synthetic_linear_task(args):
    score, cost = synthetic_basic_task(args)
    return score*cost, cost

def synthetic_log_task(args):
    score, cost = synthetic_basic_task(args)
    return score*np.log10(cost), cost

def synthetic_percentile_task(args):
    score, cost = synthetic_basic_task(args)
    return score/(1 + np.exp(-cost/10)), cost

def synthetic_cutoff_task(args):
    score, cost = synthetic_basic_task(args)
    return score*min(2, np.log10(cost)), cost

def test_sweep(args):
    method = args['sweep']['method']
    if method == 'random':
        sweep = pufferlib.sweep.Random(args['sweep'])
    elif method == 'pareto_genetic':
        sweep = pufferlib.sweep.ParetoGenetic(args['sweep'])
    elif method == 'protein':
        sweep = pufferlib.sweep.Protein(
            args['sweep'],
            resample_frequency=5,
            num_random_samples=10, # Should be number of params
            max_suggestion_cost=args['base']['max_suggestion_cost'],
            min_score = args['sweep']['metric']['min'],
            max_score = args['sweep']['metric']['max'],
        )
    else:
        raise ValueError(f'Invalid sweep method {method} (random/pareto_genetic/protein)')

    target_metric = args['sweep']['metric']['name']
    scores, costs = [], []
    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
 
        info = sweep.suggest(args)
        score, cost = synthetic_log_task(args)
        sweep.observe(score, cost)
        print('Score:', score, 'Cost:', cost)

        scores.append(score)
        costs.append(cost)

    pareto, pareto_idx = pufferlib.sweep.pareto_points(sweep.success_observations)

    pareto_scores = np.array(scores)[pareto_idx].tolist()
    pareto_costs = np.array(costs)[pareto_idx].tolist()

    np.save(args['data_path']+'.npy', {'scores': scores, 'costs': costs})
    np.save(args['data_path']+'_pareto.npy', {'scores': pareto_scores, 'costs': pareto_costs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--data-path', type=str, default='sweep',
        help='Used for testing hparam algorithms')
    parser.add_argument('--max-runs', type=int, default=200, help='Max number of sweep runs')
    parser.add_argument('--tag', type=str, default=None, help='Tag for experiment')
    parser.add_argument('--wandb', action='store_true', help='Track on WandB')
    parser.add_argument('--neptune', action='store_true', help='Track on Neptune')
    args = parser.parse_known_args()[0]

    p = configparser.ConfigParser()
    p.read('config/default.ini')
    for section in p.sections():
        for key in p[section]:
            argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    parsed = parser.parse_args().__dict__
    args = {'env': {}, 'policy': {}, 'rnn': {}}
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    test_sweep(args)
    

