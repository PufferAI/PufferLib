import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.checkers import binding

class Checkers(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=8, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(size*size,), dtype=np.uint8)
        num_move_types = 8  # Move types are: NW, NE, SW, SE, 2*NW, 2*NE, 2*SW, 2*SE,
        action_space_size = size * size * num_move_types
        self.single_action_space = gymnasium.spaces.Discrete(action_space_size)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size)
 
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

if __name__ == '__main__':
    N = 4096
    size = 3

    env = Checkers(num_envs=N, size=size)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, size * size * 6, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('Checkers SPS:', int(steps / (time.time() - start)))
