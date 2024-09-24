'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

from raylib import rl

import pufferlib
from pufferlib.environments.ocean.connect4.cy_connect4 import CyConnect4

class MyConnect4(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=128,
             width=672, height=576, piece_width=96, piece_height=96, longest_connected=0, game_over=0, pieces_placed=0):
        super().__init__()

        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # sim hparams (px, px/tick)
        self.width = width
        self.height = height
        self.piece_width = piece_width
        self.piece_height = piece_height
        self.longest_connected = longest_connected
        self.game_over = game_over
        self.pieces_placed = pieces_placed
        # spaces
        self.num_obs = 42
        self.num_act = 7
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
        self.reward_sum = 0


    def reset(self, seed=None):
        self.tick = 0
        self.c_envs = []

        for i in range(self.num_envs):
            # TODO: since single agent, could we just pass values by reference instead of (1,) array?
            self.c_envs.append(CyConnect4(self.actions[i:i+1],
                self.buf.observations[i], self.buf.rewards[i:i+1], self.buf.terminals[i:i+1],
                self.width, self.height, self.piece_width, self.piece_height, self.longest_connected, self.game_over, self.pieces_placed))
            self.c_envs[i].reset()

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        for i in range(self.num_envs):
            self.c_envs[i].step()

        # TODO: hacky way to convert uint8 to bool
        self.buf.terminals[:] = self.terminals_uint8.astype(bool)
        self.tick += 1
        info = {}
        self.reward_sum += self.buf.rewards.mean()

        if self.tick % self.report_interval == 0:
            info.update({
                'reward': self.reward_sum / self.report_interval,
            })

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        self.c_envs[0].render()

def test_performance(timeout=10, atn_cache=1024):
    env = MyConnect4(num_envs=1000)
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
