import time
import random

import numpy as np
import torch

from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import pufferlib
import pufferlib.sweep

try:
    from bokeh.models import ColumnDataSource, LinearColorMapper
    from bokeh.plotting import figure, show
    from bokeh.palettes import Turbo256
except:
    pass

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
    noise_cost = cost + 0.20*np.random.randn()*cost
    noise_cost = min(noise_cost, 200)
    noise_cost = max(noise_cost, 1)
    return score*np.log10(noise_cost), cost

def synthetic_percentile_task(args):
    score, cost = synthetic_basic_task(args)
    noise_cost = cost - 0.20*abs(np.random.randn())*cost
    noise_cost = min(noise_cost, 200)
    noise_cost = max(noise_cost, 1)
    return score/(1 + np.exp(-noise_cost/10)), cost

def synthetic_cutoff_task(args):
    score, cost = synthetic_basic_task(args)
    return score*min(2, np.log10(cost)), cost

def test_sweep(args):
    method = args['sweep']['method']
    if method == 'Random':
        sweep = pufferlib.sweep.Random(args['sweep'])
    elif method == 'ParetoGenetic':
        sweep = pufferlib.sweep.ParetoGenetic(args['sweep'])
    elif method == 'Protein':
        sweep = pufferlib.sweep.Protein(
            args['sweep'],
            expansion_rate = 1.0,
            use_gpu=True,
            prune_pareto=True,
            # ucb_beta=0.1,
        )
    else:
        raise ValueError(f'Invalid sweep method {method} (random/pareto_genetic/protein)')

    task = args['task']
    if task == 'linear':
        synthetic_task = synthetic_linear_task
    elif task == 'log':
        synthetic_task = synthetic_log_task
    elif task == 'percentile':
        synthetic_task = synthetic_percentile_task
    else:
        raise ValueError(f'Invalid task {task}')

    target_metric = args['sweep']['metric']
    scores, costs = [], []
    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
 
        start_time = time.time()
        _, info = sweep.suggest(args)
        suggestion_time = time.time() - start_time

        info_str = f'{i}th sweep.suggest() took {suggestion_time:.4f}s'
        if "score_loss" in info:
            info_str += f', score_loss: {info["score_loss"]:.4f}'
        if "cost_loss" in info:
            info_str += f', cost_loss: {info["cost_loss"]:.4f}'
        if "score_lengthscale" in info:
            info_str += f'\nscore_lengthscale: {info["score_lengthscale"]}'
        if "cost_lengthscale" in info:
            info_str += f', cost_lengthscale: {info["cost_lengthscale"]}'
        print(info_str)

        total_timesteps = args['train']['total_timesteps']
        for i in range(1, 6):
            args['train']['total_timesteps'] = i*total_timesteps/5
            score, cost = synthetic_task(args)
            sweep.observe(args, score, cost)
            print('Score:', score, 'Cost:', cost)

        scores.append(score)
        costs.append(cost)

    pareto, pareto_idx = pufferlib.sweep.pareto_points(sweep.success_observations)

    np.save(args['data_path']+'.npy', {'scores': scores, 'costs': costs})

    #pareto_scores = np.array(scores)[pareto_idx].tolist()
    #pareto_costs = np.array(costs)[pareto_idx].tolist()
    #np.save(args['data_path']+'_pareto.npy', {'scores': pareto_scores, 'costs': pareto_costs})

def visualize(args):
    data = np.load(args['vis_path'] + '.npy', allow_pickle=True).item()
    costs = data['costs']
    scores = data['scores']

    sorted_costs = np.sort(costs)
    aoc = np.max(scores) * np.cumsum(sorted_costs) / np.sum(costs)

    # Create a ColumnDataSource that includes the 'order' for each point
    source = ColumnDataSource(data=dict(
        x=costs,
        y=scores,
        order=list(range(len(scores)))  # index/order for each point
    ))

    curve = ColumnDataSource(data=dict(
        x=sorted_costs,
        y=aoc,
        order=list(range(len(scores)))  # index/order for each point
    ))

    # Define a color mapper across the range of point indices
    mapper = LinearColorMapper(
        palette=Turbo256,
        low=0,
        high=len(scores)
    )

    # Set up the figure
    p = figure(title='Synthetic Hyperparam Test', 
               x_axis_label='Cost', 
               y_axis_label='Score')

    # Use the 'order' field for color -> mapped by 'mapper'
    p.scatter(x='x', 
              y='y', 
              color={'field': 'order', 'transform': mapper}, 
              size=10, 
              source=source)

    p.line(x='x', 
           y='y', 
           color='purple',
           source=curve)

    show(p)


if __name__ == '__main__':
    from pufferlib import pufferl

    parser = pufferl.make_parser()
    parser.add_argument('--task', type=str, default='linear', help='Task to optimize')
    parser.add_argument('--vis-path', type=str, default='',
        help='Set to visualize a saved sweep')
    parser.add_argument('--data-path', type=str, default='sweep',
        help='Used for testing hparam algorithms')

    args = pufferl.load_config('default', parser=parser)

    if args['vis_path']:
        visualize(args)
        exit(0)

    test_sweep(args)
