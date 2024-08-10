import numpy as np
import torch
import time

import pufferlib
import random
import glob
import os
from pufferlib.policy_ranker import update_elos

def load_policy(checkpoint_dir):
    paths = glob.glob(f'{checkpoint_dir}/model_*.pt', recursive=True)
    path = random.choice(paths)
    return torch.load(path, map_location='cpu'), path.split('/')[-1]

def init_elos(checkpoint_dir):
    paths = glob.glob(f'{checkpoint_dir}/model_*.pt', recursive=True)
    elos = {path.split('/')[-1]: 1000 for path in paths}
    elos['mine'] = 1000
    return elos

def rollout(env, policy, opponent, timeout=300, render=False):
    obs, _ = env.reset()
    cenv = env.c_envs[0]

    my_state = None
    opp_state = None

    start = time.time()
    step = 0

    prev_radiant_victories = cenv.radiant_victories
    prev_dire_victories = cenv.dire_victories
    prev_towers_taken = 0
    prev_taken_step = 0

    while time.time() - start < timeout:
        towers_taken = cenv.total_towers_taken
        if towers_taken > prev_towers_taken:
            prev_towers_taken = towers_taken
            prev_taken_step = step
        elif step - prev_taken_step > 50000:
            return None

        if step % 1000 == 0:
            print(
                f'Step: {step} Radiant Victories: {cenv.radiant_victories}'
                f' Dire Victories: {cenv.dire_victories}'
                f' Towers Taken: {cenv.total_towers_taken}'
            )

        if render and step % 10 == 0:
            env.render()

        step += 1
        with torch.no_grad():
            obs = torch.as_tensor(obs)
            my_obs = obs[:5]
            opp_obs = obs[5:]
            if hasattr(policy, 'lstm'):
                my_action, _, _, _, my_state = policy(my_obs, my_state)
                opp_action, _, _, _, opp_state = opponent(opp_obs, opp_state)
            else:
                my_action, _, _, _ = policy(my_obs)
                opp_action, _, _, _ = opponent(opp_obs)

        action = torch.cat([my_action, opp_action])
        obs, reward, done, truncated, info = env.step(action.numpy())

        if cenv.radiant_victories > prev_radiant_victories:
            prev_radiant_victories = cenv.radiant_victories
            print('Radiant Victory')
            return np.array([1, 0])
        elif cenv.dire_victories > prev_dire_victories:
            prev_dire_victories = cenv.dire_victories
            print('Dire Victory')
            return np.array([0, 1])
        else:
            continue

    return None

if __name__ == '__main__':
    from pufferlib.environments.ocean.environment import env_creator
    from pufferlib.environments.ocean.torch import MOBA, Recurrent
    import pufferlib.frameworks.cleanrl

    checkpoint_dir = 'moba_checkpoints'

    make_env = env_creator('moba')
    env = make_env(num_envs=1)#, render_mode='raylib')

    policy = MOBA(env)
    policy = Recurrent(env, policy)
    policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)

    elos = init_elos(checkpoint_dir)
    for game in range(100):
        opponent, name = load_policy(checkpoint_dir)
        print(f'Game: {game} Opponent: {name}')
        scores = rollout(env, policy, opponent, render=False)
        if scores is None:
            continue

        elo_pair = np.array([elos['mine'], elos[name]])
        elo_pair = update_elos(elo_pair, scores)
        elos['mine'] = elo_pair[0]
        elos[name] = elo_pair[1]

        for k, v in elos.items():
            print(f'{k}: {v}')
        print()




