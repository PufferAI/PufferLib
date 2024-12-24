# enduro_clone.py

import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.enduro.cy_enduro import CyEnduro

class Enduro(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, frame_skip=1, render_mode='human',
                 report_interval=1, buf=None):

        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.frame_skip = frame_skip
        self.tick = 0
        self.max_enemies = 10

        obs_size = (8 + (5 * self.max_enemies) + 9 + 1)
        self.num_obs = obs_size

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(self.num_obs,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(9)

        # Initialize buffers
        self.observations = np.zeros((self.num_agents, self.num_obs), dtype=np.float32)
        self.actions = np.zeros((self.num_agents,), dtype=np.int32)
        self.rewards = np.zeros((self.num_agents,), dtype=np.float32)
        self.terminals = np.zeros((self.num_agents,), dtype=np.uint8)
        self.truncations = np.zeros((self.num_agents,), dtype=np.uint8)

        # Rewards buffer for smoothing reward curve
        self.rewards_buffer = []

        super().__init__(buf=buf)

        self.c_envs = CyEnduro(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs
        )

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        for _ in range(self.frame_skip):
            self.actions[:] = actions
            self.c_envs.step()

            # Check for terminal or truncated states
            if np.any(self.terminals) or np.any(self.truncations):
                break

        # Update rewards buffer
        self.rewards_buffer.append(np.mean(self.rewards))

        info = []
        if self.tick % self.report_interval == 0:
            rewards = np.mean(self.rewards_buffer)
            info.append({'rewards': rewards})
            self.rewards_buffer = []
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        self.tick += 1

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

    def validate_probabilities(prob_tensor):
        import torch
        if torch.isnan(prob_tensor).any() or torch.isinf(prob_tensor).any() or (prob_tensor < 0).any():
            raise ValueError("Invalid probability values detected")
        return prob_tensor

def test_performance(timeout=10, atn_cache=8192):
    num_envs = 4096
    env = Enduro(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')

if __name__ == '__main__':
    test_performance()
