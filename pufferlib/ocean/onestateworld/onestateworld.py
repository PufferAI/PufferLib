'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.onestateworld import binding

class World(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, 
                 mean_left=0.1, mean_right=1, var_right=5, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=0,
            shape=(1,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, 
            mean_left=mean_left, mean_right=mean_right, var_right=var_right
        )
 
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
    size = 10

    env = World(size=size)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 2, (CACHE,))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += 1
        i += 1

    print('SPS:', int(steps / (time.time() - start)))
