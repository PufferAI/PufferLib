import numpy as np
import torch
import time

import pufferlib
import random
import glob
import os
from pufferlib.policy_ranker import update_elos

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

def rollout(envs, policy, opponents, num_games, timeout=300, render=False):
    obs, _ = envs.reset()
    #cenv = envs.c_envs[0]

    start = time.time()
    step = 0

    num_envs = len(envs.c_envs)
    num_opponents = len(opponents)
    envs_per_opponent = num_envs // num_opponents
    my_state = None
    opp_states = [None for _ in range(num_opponents)]
    prev_radiant_victories = [c.radiant_victories for c in envs.c_envs]
    prev_dire_victories = [c.dire_victories for c in envs.c_envs]

    scores = []

    slice_idxs = torch.arange(10*num_envs).reshape(num_envs, 10).cuda()
    my_idxs = slice_idxs[:, :5].reshape(num_envs*5)
    opp_idxs = slice_idxs[:, 5:].reshape(num_envs*5).split(5*envs_per_opponent)

    games_played = 0
    while games_played < num_games:
        #if render and step % 10 == 0:
        #    env.render()

        step += 1
        opp_actions = []
        with torch.no_grad():
            obs = torch.as_tensor(obs).cuda()
            my_obs = obs[my_idxs]

            # Parallelize across opponents
            if hasattr(policy, 'lstm'):
                my_actions, _, _, _, my_state = policy(my_obs, my_state)
            else:
                my_actions, _, _, _ = policy(my_obs)

            # Iterate opponent policies
            for i in range(num_opponents):
                opp_obs = obs[opp_idxs[i]]
                opp_state = opp_states[i]

                opponent = opponents[i]
                if hasattr(policy, 'lstm'):
                    opp_atn, _, _, _, opp_states[i] = opponent(opp_obs, opp_state)
                else:
                    opp_atn, _, _, _ = opponent(opp_obs)

                opp_actions.append(opp_atn)

        opp_actions = torch.cat(opp_actions)
        actions = torch.cat([
            my_actions.view(num_envs, 5, -1),
            opp_actions.view(num_envs, 5, -1),
        ], dim=1).view(num_envs*10, -1)

        obs, reward, done, truncated, info = envs.step(actions.cpu().numpy())

        for i in range(num_envs):
            c = envs.c_envs[i]
            if c.radiant_victories > prev_radiant_victories[i]:
                prev_radiant_victories[i] = c.radiant_victories
                scores.append((i, 1))
                games_played += 1
                print('Radiant Victory')
            elif c.dire_victories > prev_dire_victories[i]:
                prev_dire_victories[i] = c.dire_victories
                scores.append((i, 0))
                games_played += 1
                print('Dire Victory')

    return scores

if __name__ == '__main__':
    from pufferlib.environments.ocean.environment import env_creator
    from pufferlib.environments.ocean.torch import MOBA, Recurrent
    import pufferlib.frameworks.cleanrl

    checkpoint_dir = 'moba_checkpoints'
    n = 8

    make_env = env_creator('moba')
    envs = make_env(num_envs=64)#, render_mode='raylib')

    #policy = MOBA(env)
    #policy = Recurrent(env, policy)
    #policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    policy = torch.load(os.path.join(checkpoint_dir, 'model_000476.pt'), map_location='cuda')
    opponents = load_policies(checkpoint_dir, n=8)

    paths = glob.glob(f'{checkpoint_dir}/model_*.pt', recursive=True)
    names = [path.split('/')[-1] for path in paths]
    elos = {name: 1000 for name in names}
    elos['mine'] = 1000
    
    # Sample with replacement if not enough models
    n_models = len(paths)
    if n_models < n:
        idxs = random.choices(range(n_models), k=n)
    else:
        idxs = random.sample(range(n_models), n)

    opponent_names = [names[i] for i in idxs]
    opponents = [torch.load(paths[i], map_location='cuda') for i in idxs]

    results = rollout(envs, policy, opponents, num_games=16, render=False)

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




