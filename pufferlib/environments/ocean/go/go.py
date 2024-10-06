'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

from raylib import rl

import pufferlib
from pufferlib.environments.ocean.go.cy_go import CyGo

class MyGo(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            width=1500, height=1200,
            grid_size=18,
            board_width=1000, board_height=1000,
            grid_square_size=1000/18,
            moves_made=0,
            komi=7.5):
        super().__init__()

        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # sim hparams (px, px/tick)
        self.grid_size = grid_size
        self.board_width = board_width
        self.board_height = board_height
        self.grid_square_size = grid_square_size
        self.moves_made = moves_made
        self.width = width
        self.height = height
        self.komi = komi
        
        # misc logging
        self.reward_sum = 0.0
        self.num_games = 0
        # spaces
        self.num_obs = (self.grid_size+1) * (self.grid_size+1) + 1
        self.num_act = (self.grid_size+1) * (self.grid_size+1) + 1
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
            self.c_envs.append(CyGo(self.actions[i:i+1],
                self.buf.observations[i], self.buf.rewards[i:i+1], self.buf.terminals[i:i+1],
                self.width, self.height, self.grid_size, self.board_width, self.board_height, self.grid_square_size, self.moves_made, self.komi))
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
        self.num_games += int(np.sum(self.buf.terminals))  # Convert to int
        if self.tick % self.report_interval == 0:

            info.update({
                'reward': self.reward_sum / self.report_interval,
                'num_games': self.num_games,
            })
            self.reward_sum = 0.0

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        self.c_envs[0].render()

def test_performance(timeout=10, atn_cache=1024):
    env = MyGo(num_envs=1000)
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
