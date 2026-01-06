'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

from pufferlib.ocean.slimevolley import binding
import pufferlib

class SlimeVolley(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0,
                 num_agents=1):
        assert num_agents in {1, 2}, "num_agents must be 1 or 2"
        num_obs = 12
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([2, 2, 2])

        self.render_mode = render_mode
        self.num_agents = num_envs * num_agents
        self.log_interval = log_interval

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                seed,
                num_agents=num_agents
                )
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
    N = 8

    env = SlimeVolley(num_envs=N, num_agents=2)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.nvec, size=(CACHE, N*2, 3))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('SlimeVolley SPS:', int(steps / (time.time() - start)))