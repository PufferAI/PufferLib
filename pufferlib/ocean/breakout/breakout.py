'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.breakout.cy_breakout import CyBreakout

class Breakout(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=128,
            frameskip=4, width=576, height=330,
            paddle_width=62, paddle_height=8,
            ball_width=32, ball_height=32,
            brick_width=32, brick_height=12,
            brick_rows=6, brick_cols=18, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(11 + brick_rows*brick_cols,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(3)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = CyBreakout(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, frameskip, width, height,
            paddle_width, paddle_height, ball_width, ball_height,
            brick_width, brick_height, brick_rows, brick_cols)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    env = CyBreakout(num_envs=1000)
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

if __name__ == '__main__':
    test_performance()
