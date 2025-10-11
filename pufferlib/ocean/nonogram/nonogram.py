'''Nonogram logic puzzle environment'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.nonogram import binding

class Nonogram(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=8, buf=None, seed=0):
        max_clues = size // 2
        obs_size = size * size + 2 * size * max_clues
        self.size = size

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=size,
            shape=(obs_size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(size * size)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size)

        # Allocate array for solutions
        self.solutions = np.zeros((num_envs, size * size), dtype=np.uint8)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

    def get_solutions(self):
        """Get the solution grids for all environments"""
        binding.vec_get_solutions(self.c_envs, self.solutions)
        return self.solutions

if __name__ == '__main__':
    N = 4096

    env = Nonogram(num_envs=N, size=8)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 64, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('Nonogram SPS:', int(steps / (time.time() - start)))
