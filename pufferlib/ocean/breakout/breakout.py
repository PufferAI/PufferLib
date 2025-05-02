'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.breakout import binding

class Breakout(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
            frameskip=4, width=576, height=330,
            paddle_width=62, paddle_height=8,
            ball_width=32, ball_height=32,
            brick_width=32, brick_height=12,
            brick_rows=6, brick_cols=18, continuous=False, log_interval=128,
            buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(10 + brick_rows*brick_cols,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.continuous = continuous
        self.log_interval = log_interval
        self.tick = 0
        
        if continuous:
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1,
                shape=(1,), dtype=np.float32)
        else:
            self.single_action_space = gymnasium.spaces.Discrete(3)
            
        super().__init__(buf)
        if continuous:
            self.actions = self.actions.flatten()
        else:
            self.actions = self.actions.astype(np.float32)
            
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, frameskip=frameskip, width=width, height=height,
            paddle_width=paddle_width, paddle_height=paddle_height, ball_width=ball_width, ball_height=ball_height,
            brick_width=brick_width, brick_height=brick_height, brick_rows=brick_rows,
            brick_cols=brick_cols, continuous=continuous
        )

    def reset(self, seed=None):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        if self.continuous:
            self.actions[:] = np.clip(actions.flatten(), -1.0, 1.0)
        else:
            self.actions[:] = actions
            
        self.tick += 1
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

def test_performance(timeout=10, atn_cache=1024):
    env = Breakout(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
