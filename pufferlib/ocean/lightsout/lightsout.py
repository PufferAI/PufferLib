"""Scaffold for a future LightsOut ocean environment."""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.lightsout import binding

import time

class LightsOut(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, grid_size=5, max_steps=None, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(grid_size * grid_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(grid_size * grid_size)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0
        
        if max_steps is None:
            max_steps = grid_size * grid_size * 10

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            grid_size=grid_size,
            cell_size=int(np.ceil(1280 / grid_size)),
            max_steps=max_steps,
        )
        self.grid_size = grid_size

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
          seed = time.time_ns() & 0x7FFFFFFF
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


if __name__ == "__main__":
    n = 4096
    env = LightsOut(num_envs=n)
    env.reset()
    steps = 0

    cache = 1024
    actions = np.zeros((cache, n), dtype=np.int32)

    import time

    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % cache])
        steps += 1

    print("LightsOut SPS:", int(env.num_agents * steps / (time.time() - start)))
