'''Action0 - A simple credit assignment test environment.

The agent must take action 0 on the first step to win.
The episode terminates at step 128 and reward is given only at termination.
'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.action0 import binding


class Action0(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128,
                 horizon=128, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            horizon=horizon)

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
    N = 4096

    env = Action0(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 2, (CACHE, N))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print('Action0 SPS:', int(env.num_agents * steps / (time.time() - start)))
