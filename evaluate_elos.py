import numpy as np
import torch
import time

import pufferlib
import random
import glob
import os

from pufferlib.policy_ranker import update_elos
from pufferlib.environments.ocean.environment import env_creator
from pufferlib.environments.ocean.torch import MOBA, Recurrent
import pufferlib.frameworks.cleanrl


def load_policies(checkpoint_dir, n, map_location='cuda'):
    paths = glob.glob(f'{checkpoint_dir}/model_*.pt', recursive=True)

    # Sample with replacement if not enough models
    if len(paths) < n:
        samples = random.choices(paths, k=n)
    else:
        samples = random.sample(paths, n)

    names = [path.split('/')[-1] for path in samples]
    return {name: torch.load(path, map_location=map_location)
        for name, path in zip(names, samples)}

def rollout(envs, policy, opponents, num_games, timeout=180, render=False):
    obs, _ = envs.reset()

    # Double reset clears randomizations
    obs, _ = envs.reset()
    #cenv = envs.c_envs[0]

    start = time.time()
    step = 0

    num_envs = len(envs.c_envs)
    num_opponents = len(opponents)
    envs_per_opponent = num_envs // num_opponents
    my_states = [None for _ in range(num_opponents)]
    opp_states = [None for _ in range(num_opponents)]
    prev_radiant_victories = [c.radiant_victories for c in envs.c_envs]
    prev_dire_victories = [c.dire_victories for c in envs.c_envs]
    scores = []

    atn_shape = (10*num_envs, len(envs.action_space.nvec))
    actions = torch.zeros(atn_shape, dtype=torch.int64).cuda()
    actions_struct = actions.view(num_opponents, envs_per_opponent, 2, 5, len(envs.action_space.nvec))

    slice_idxs = torch.arange(10*num_envs).reshape(num_opponents, envs_per_opponent, 2, 5).cuda()
    flat_teams = np.random.randint(0, 2, num_envs)
    team_assignments = torch.from_numpy(flat_teams.reshape(num_opponents, envs_per_opponent)).cuda()
    arange = torch.arange(envs_per_opponent).cuda()

    games_played = 0
    while games_played < num_games and time.time() - start < timeout:
        #if render and step % 10 == 0:
        #    env.render()

        step += 1
        with torch.no_grad():
            obs = torch.as_tensor(obs).cuda()
            for i in range(num_opponents):
                idxs = slice_idxs[i]
                teams = team_assignments[i]

                my_obs = obs[idxs[arange, teams]].view(5*envs_per_opponent, -1)
                opp_obs = obs[idxs[arange, 1 - teams]].view(5*envs_per_opponent, -1)

                if hasattr(policy, 'lstm'):
                    my_actions, _, _, _, my_states[i] = policy(my_obs, my_states[i])
                    opp_atn, _, _, _, opp_states[i] = opponents[i](opp_obs, opp_states[i])
                else:
                    my_actions, _, _, _ = policy(my_obs)
                    opp_atn, _, _, _ = opponents[i](opp_obs)

                actions_struct[i, arange, teams] = my_actions.view(envs_per_opponent, 5, -1)
                actions_struct[i, arange, 1 - teams] = opp_atn.view(envs_per_opponent, 5, -1)

        obs, reward, done, truncated, info = envs.step(actions.cpu().numpy())

        for i in range(num_envs):
            c = envs.c_envs[i]
            opp_idx = i // envs_per_opponent
            if c.radiant_victories > prev_radiant_victories[i]:
                prev_radiant_victories[i] = c.radiant_victories
                scores.append((opp_idx, flat_teams[i] == 0))
                games_played += 1
                print('Radiant Victory')
            elif c.dire_victories > prev_dire_victories[i]:
                prev_dire_victories[i] = c.dire_victories
                scores.append((opp_idx, flat_teams[i] == 1))
                games_played += 1
                print('Dire Victory')

    return scores

def calc_elo(checkpoint, checkpoint_dir, elos, num_envs=128, num_games=128, num_opponents=8, k=24.0):
    print(f'Calculating ELO for {checkpoint}')
    make_env = env_creator('moba')
    envs = make_env(num_envs=num_envs)

    policy = torch.load(os.path.join(checkpoint_dir, checkpoint), map_location='cuda')
    print(f'Loaded policy {checkpoint}')

    paths = glob.glob(f'{checkpoint_dir}/model_*.pt', recursive=True)
    names = [path.split('/')[-1] for path in paths]
    print(f'Loaded {len(paths)} models')
    paths.remove(f'{checkpoint_dir}/{checkpoint}')
    print(f'Removed {checkpoint} from paths')
    elos[checkpoint] = 1000
    
    # Sample with replacement if not enough models
    print(f'Sampling {num_opponents} opponents')
    n_models = len(paths)
    if n_models < num_opponents:
        idxs = random.choices(range(n_models), k=num_opponents)
    else:
        idxs = random.sample(range(n_models), num_opponents)
    print(f'Sampled {num_opponents} opponents')

    opponent_names = [names[i] for i in idxs]
    opponents = [torch.load(paths[i], map_location='cuda') for i in idxs]
    print(f'Loaded {num_opponents} opponents')

    results = rollout(envs, policy, opponents, num_games=num_games, render=False)
    print(f'Finished {num_games} games')

    for game in results:
        opponent, win = game
        if win:
            score = np.array([1, 0])
        else:
            score = np.array([0, 1])

        opp_name = opponent_names[opponent]
        elo_pair = np.array([elos[checkpoint], elos[opp_name]])
        elo_pair = update_elos(elo_pair, score, k=24.0)
        elos[checkpoint] = elo_pair[0]
        #elos[opp_name] = elo_pair[1]

    print(f'Finished calculating ELO for {checkpoint}')
    for k, v in elos.items():
        print(f'{k}: {v}')

    return elos

    '''
    for game in range(1000):
        opponent, name = load_policy(checkpoint_dir)
        print(f'Game: {game} Opponent: {name}')
        scores = rollout(env, policy, opponent, render=False)
        if scores is None:
            continue

        elo_pair = np.array([elos['mine'], elos[name]])
        elo_pair = update_elos(elo_pair, scores, k=24.0)
        elos['mine'] = elo_pair[0]
        elos[name] = elo_pair[1]

        for k, v in elos.items():
            print(f'{k}: {v}')
        print()
    '''


if __name__ == '__main__':
    checkpoint_dir = 'moba_elo'
    checkpoint = 'model_0.pt'
    elos = {'model_random.pt': 1000}
    calc_elo(checkpoint, checkpoint_dir, elos, num_games=16)

