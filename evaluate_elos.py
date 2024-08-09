import numpy as np
import torch
import time

import pufferlib
import random
import glob
import os
from pufferlib.policy_ranker import update_elos

def load_policy(checkpoint_dir):
    paths = glob.glob(f'{checkpoint_dir}/*.pt', recursive=True)
    path = random.choice(paths)
    return torch.load(path, map_location='cpu')

class EloMOBA:
    def __init__(self):
        self.prev_radiant_victories = 0
        self.prev_dire_victories = 0
        self.elos = 1000 + np.zeros(2)

    def rollouts(self, env, policy, checkpoint_dir, timeout=60):
        obs, _ = env.reset()
        cenv = env.c_envs[0]
        my_state = None
        opp_state = None
        games_played = 0
        opponent = load_policy(checkpoint_dir)

        step = 0
        start = time.time()
        while time.time() - start < timeout:
            if step % 100 == 0:
                print(f'Step: {step} Radiant Victories: {cenv.radiant_victories} Dire Victories: {cenv.dire_victories} Towers Taken: {cenv.total_towers_taken}')

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

            if cenv.radiant_victories > self.prev_radiant_victories:
                self.prev_radiant_victories = cenv.radiant_victories
                scores = np.array([1, 0])
            elif cenv.dire_victories > self.prev_dire_victories:
                self.prev_dire_victories = cenv.dire_victories
                scores = np.array([0, 1])
            else:
                continue

            games_played += 1
            update_elos(self.elos, scores)
            opponent = load_policy(checkpoint_dir)
            print(elos)

        return self.elos

if __name__ == '__main__':
    from pufferlib.environments.ocean.environment import env_creator
    from pufferlib.environments.ocean.torch import MOBA, Recurrent
    import pufferlib.frameworks.cleanrl

    make_env = env_creator('moba')
    env = make_env(num_envs=1)

    policy = MOBA(env)
    policy = Recurrent(env, policy)
    policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)

    elo_evaluator = EloMOBA()
    elos = elo_evaluator.rollouts(env, policy, 'moba_checkpoints')











