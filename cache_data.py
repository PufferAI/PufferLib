import numpy as np

import json
import glob
import os


env_names = sorted([
    'breakout',
    #'impulse_wars',
    #'pacman',
    #'tetris',
    #'g2048',
    #'moba',
    #'pong',
    #'tower_climb',
    #'grid',
    #'nmmo3',
    #'snake',
    #'tripletriad'
])

HYPERS = [
    'train/learning_rate',
    'train/ent_coef',
    'train/gamma',
    'train/gae_lambda',
    'train/vtrace_rho_clip',
    'train/vtrace_c_clip',
    'train/clip_coef',
    'train/vf_clip_coef',
    'train/vf_coef',
    'train/max_grad_norm',
    'train/beta1',
    'train/beta2',
    'train/eps',
    'train/prio_alpha',
    'train/prio_beta0',
    'train/horizon',
    'train/replay_ratio',
    'train/minibatch_size',
    'policy/hidden_size',
    'vec/total_agents',
]

ALL_KEYS = [
    'agent_steps',
    'cost',
    'environment/score',
    'environment/perf'
] + HYPERS

def pareto_idx(steps, costs, scores):
    idxs = []
    for i in range(len(steps)):
        better = [scores[j] >= scores[i] and
            costs[j] < costs[i] and steps[j] < steps[i]
            for j in range(len(scores))]
        if not any(better):
            idxs.append(i)

    return idxs

def load_sweep_data(path):
    data = {}
    keys = None
    for fpath in glob.glob(path):
        if 'cache.json' in fpath:
            continue

        with open(fpath, 'r') as f:
            exp = json.load(f)

        if not data:
            for kk in exp.keys():
                if kk == 'data':
                    for k, v in exp[kk][-1].items():
                        data[k] = []
                else:
                    data[kk] = []

        discard = False
        for kk in list(data.keys()):
            if kk not in exp and kk not in exp['data'][-1]:
                discard = True
                break

        if discard:
            continue

        for kk in list(data.keys()):
            if kk in exp:
                v = exp[kk]
                sweep_key = f'sweep/{kk}/distribution'
                if sweep_key in data and exp[sweep_key] == 'logit_normal':
                    v = 1 - v
                elif kk in ('train/vtrace_rho_clip', 'train/vtrace_c_clip'):
                    v = max(v, 0.1)

                data[kk].append(v)
            else:
                data[kk].append(exp['data'][-1][kk])

    steps = data['agent_steps']
    costs = data['cost']
    scores = data['environment/score']

    idxs = pareto_idx(steps, costs, scores)

    # Filter to pareto
    for k in data:
        data[k] = [data[k][i] for i in idxs]

    # Monkey patch: Cap performance
    data['environment/perf'] = [min(e, 1.0) for e in data['environment/perf']]
    
    # Monkey patch: Adjust steps by frameskip if present
    if 'env/frameskip' in data:
        skip = data['env/frameskip']
        data['agent_steps'] = [n*m for n, m in zip(data['agent_steps'], skip)]
 
    return data

def cached_sweep_load(path, env_name):
    cache_file = os.path.join(path, 'c_cache.json')
    if not os.path.exists(cache_file):
        data = load_sweep_data(os.path.join(path, '*.json'))
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    with open(cache_file, 'r') as f:
        data = json.load(f)

    print(f'Loaded {env_name}')
    return data

def compute_tsne():
    data = {name: cached_sweep_load(f'experiments/logs/puffer_{name}', name) for name in env_names}

    flat = []
    flat_mmin = []
    flat_mmax = []
    for env in env_names:
        flat.append(np.stack([data[env][hyper] for hyper in HYPERS], axis=1))
        flat_mmin.append(np.stack([data[env][f'sweep/{hyper}/min'] for hyper in HYPERS], axis=1))
        flat_mmax.append(np.stack([data[env][f'sweep/{hyper}/max'] for hyper in HYPERS], axis=1))

    flat_distribution = [data[env][f'sweep/{hyper}/distribution'] for env in env_names for hyper in HYPERS]

    flat = np.concatenate(flat, axis=0)
    flat_mmin = np.concatenate(flat_mmin, axis=0).min(axis=0)
    flat_mmax = np.concatenate(flat_mmax, axis=0).max(axis=0)

    normed = flat.copy()
    for i in range(len(HYPERS)):
        dist = flat_distribution[i]
        if 'log' in dist or 'pow2' in dist:
            flat_mmin[i] = np.log(flat_mmin[i])
            flat_mmax[i] = np.log(flat_mmax[i])
            normed[:, i] = np.log(flat[:, i])

        normed[:, i] = (normed[:, i] - flat_mmin[i]) / (flat_mmax[i] - flat_mmin[i])

    from sklearn.manifold import TSNE
    proj = TSNE(n_components=2)
    reduced = None
    try:
        reduced = proj.fit_transform(normed)
    except ValueError:
        print('Warning: TSNE failed. Skipping TSNE')

    row = 0
    for env in env_names:
        '''
        for i, hyper in enumerate(HYPERS):
            sz = len(data[env][hyper])
            data[env][hyper] = normed[row:row+sz, i].tolist()
        '''
        sz = len(data[env]['agent_steps'])

        data[env] = {k: v for k, v in data[env].items() if k in ALL_KEYS}
        if reduced is not None:
            data[env]['tsne1'] = reduced[row:row+sz, 0].tolist()
            data[env]['tsne2'] = reduced[row:row+sz, 1].tolist()
        else:
            data[env]['tsne1'] = np.random.rand(sz).tolist()
            data[env]['tsne2'] = np.random.rand(sz).tolist()

        row += sz
        print(f'Env {env} has {sz} points')

    json.dump(data, open('all_cache.json', 'w'))

if __name__ == '__main__':
    compute_tsne()
