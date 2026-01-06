'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.battle import binding

class Battle(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=1920, height=1080, size_x=1.0,
            size_y=1.0, size_z=1.0, num_agents=1024, num_factories=32,
            num_armies=4, render_mode=None, log_interval=128, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(num_armies*3 + 4*16 + 22 + 8,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
                low=-1, high=1, shape=(3,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs*num_agents
        self.log_interval = log_interval

        if num_armies < 1 or num_armies > 8:
            raise pufferlib.APIUsageError('num_armies must be in [1, 8]')
        if num_agents % num_armies != 0:
            raise pufferlib.APIUsageError('num_agents must be a multiple of num_armies')

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                seed, width=width, height=height, size_x=size_x, size_y=size_y, size_z=size_z,
                num_agents=num_agents*2, num_factories=num_factories,
                num_armies=num_armies)
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

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
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 512

    env = Battle(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.nvec, size=(CACHE, 2))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Battle SPS:', int(steps / (time.time() - start)))
