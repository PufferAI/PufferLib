"""
FlappyBirdMini: a simple 2-block tall grid environment.
Agent can move up or down. Observes an 8x2 grid with moving obstacles.
+1 reward per obstacle passed.
"""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.flappy_bird_mini import binding

class FlappyBirdMini(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(16,), dtype=np.uint8) # 8x2 grid flattened = 16 observations
        self.single_action_space = gymnasium.spaces.Discrete(3) # 0: NOOP, 1: UP, 2: DOWN
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed)
 
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

    env = FlappyBirdMini(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 3, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('FlappyBirdMini SPS:', int(steps / (time.time() - start)))
