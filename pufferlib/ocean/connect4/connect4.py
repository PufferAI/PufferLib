'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.connect4.cy_connect4 import CyConnect4


class Connect4(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=128,
             width=672, height=576, piece_width=96, piece_height=96, buf=None, seed=0):

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(42,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(7)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf=buf)
        self.c_envs = CyConnect4(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, width, height, piece_width, piece_height)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()


def test_performance(timeout=10, atn_cache=1024, num_envs=1024):
    import time

    env = Connect4(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(
        0,
        env.single_action_space.n + 1,
        (atn_cache, num_envs),
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]         
        env.step(atn)
        tick += 1

    print(f'SPS: {num_envs * tick / (time.time() - start)}')


if __name__ == '__main__':
    test_performance()
