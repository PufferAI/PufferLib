import numpy as np
import os

import gymnasium

import pufferlib
from pufferlib.ocean.grid import binding

class Grid(pufferlib.PufferEnv):
    def __init__(self, render_mode='raylib', vision_range=5,
            num_envs=4096, num_maps=1000, map_size=-1, max_size=9,
            report_interval=128, buf=None, seed=0):
        assert map_size <= max_size
        self.obs_size = 2*vision_range + 1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        super().__init__(buf=buf)
        self.float_actions = np.zeros_like(self.actions).astype(np.float32)
        self.c_state = binding.shared(num_maps=num_maps, max_size=max_size, size=map_size)
        self.c_envs = binding.vec_init(self.observations, self.float_actions,
            self.rewards, self.terminals, self.truncations, num_envs, seed,
            state=self.c_state, max_size=max_size, num_maps=num_maps)
        pass

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.float_actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self, overlay=0):
        binding.vec_render(self.c_envs, overlay)

    def close(self):
        binding.vec_close(self.c_envs)

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
