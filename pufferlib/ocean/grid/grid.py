import numpy as np
import os

import gymnasium

import pufferlib
from pufferlib.ocean.grid.cy_grid import CGrid

class PufferGrid(pufferlib.PufferEnv):
    def __init__(self, render_mode='raylib', vision_range=5,
            num_envs=4096, num_maps=1000, max_map_size=9,
            report_interval=128, buf=None):
        self.obs_size = 2*vision_range + 1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        super().__init__(buf=buf)
        self.float_actions = np.zeros_like(self.actions).astype(np.float32)
        self.c_envs = CGrid(self.observations, self.float_actions,
            self.rewards, self.terminals, num_envs, num_maps, max_map_size)
        pass

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
               info.append(log)

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    env = CGrid(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_envs * tick / (time.time() - start))
