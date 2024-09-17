'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

from raylib import rl

import pufferlib
from pufferlib.environments.ocean.breakout.cy_breakout import CyBreakout

class MyBreakout(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=128,
            frameskip=4, width=576, height=330,
            paddle_width=62, paddle_height=8,
            ball_width=6, ball_height=6,
            brick_width=32, brick_height=12,
            brick_rows=6, brick_cols=18):
        super().__init__()

        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # sim hparams (px, px/tick)
        self.frameskip = frameskip
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width 
        self.ball_height = ball_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols

        # spaces
        self.num_obs = 11 + brick_rows*brick_cols
        self.num_act = 4
        self.observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.action_space = gymnasium.spaces.Discrete(self.num_act)
        self.single_action_space = self.action_space
        self.human_action = None

        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations = np.zeros((self.num_agents, self.num_obs,), dtype=np.float32),
            rewards = np.zeros(self.num_agents, dtype=np.float32),
            terminals = np.zeros(self.num_agents, dtype=np.bool),
            truncations = np.zeros(self.num_agents, dtype=bool),
            masks = np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint32)
        self.terminals_uint8 = np.zeros(self.num_agents, dtype=np.uint8)

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs = []

        for i in range(self.num_envs):
            # TODO: since single agent, could we just pass values by reference instead of (1,) array?
            self.c_envs.append(CyBreakout(self.frameskip, self.actions[i:i+1],
                self.buf.observations[i], self.buf.rewards[i:i+1], self.buf.terminals[i:i+1],
                self.width, self.height, self.paddle_width, self.paddle_height,
                self.ball_width, self.ball_height, self.brick_width, self.brick_height,
                self.brick_rows, self.brick_cols))
            self.c_envs[i].reset()

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        for i in range(self.num_envs):
            self.c_envs[i].step()

        # TODO: hacky way to convert uint8 to bool
        self.buf.terminals[:] = self.terminals_uint8.astype(bool)
        self.tick += 1

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, {})

    def render(self):
        self.c_envs[0].render()

def test_performance(timeout=10, atn_cache=1024):
    env = MyPong(num_envs=1000)
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
