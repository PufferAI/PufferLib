import numpy as np
import os

import gymnasium

from raylib import rl, colors

import pufferlib
from pufferlib.environments.ocean import render
from pufferlib.environments.ocean.grid.cy_grid import CGrid

EMPTY = 0
FOOD = 1
WALL = 2
AGENT_1 = 3
AGENT_2 = 4
AGENT_3 = 5
AGENT_4 = 6

PASS = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4


class PufferGrid(pufferlib.PufferEnv):
    def __init__(self, render_mode='rgb_array', vision_range=3, num_envs=4096, report_interval=1024):
        super().__init__()
        self.vision_range = vision_range

        self.obs_size = 2*self.vision_range + 1
        self.report_interval = report_interval
        self.emulated = None

        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (num_envs, self.obs_size*self.obs_size + 3), dtype=np.uint8),
            rewards = np.zeros(num_envs, dtype=np.float32),
            terminals = np.zeros(num_envs, dtype=bool),
            truncations = np.zeros(num_envs, dtype=bool),
            masks = np.ones(num_envs, dtype=bool),
        )
        self.actions = np.zeros(num_envs, dtype=np.uint32)
        self.dones = np.ones(num_envs, dtype=bool)
        self.not_done = np.zeros(num_envs, dtype=bool)

        self.render_mode = render_mode
        self.observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size+3,), dtype=np.uint8)

        self.action_space = gymnasium.spaces.Discrete(5)

        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.cenv = None
        self.done = True
        self.human_action = None
        self.infos = {}
        self.num_agents = num_envs

    def render(self):
        self.cenv.render()

    def reset(self, seed=0):
        if self.cenv is None:
            self.cenv = CGrid(self.buf.observations, self.actions,
                self.buf.rewards, self.dones, self.num_agents)
        self.cenv.reset()

        self.agents = [0]
        self.done = False
        self.tick = 1
        self.episode_reward = 0
        self.sum_rewards = 0

        return self.buf.observations, self.infos

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        self.buf.rewards.fill(0)
        self.cenv.step()

        self.sum_rewards += self.buf.rewards.sum()

        infos = {}
        if self.tick % self.report_interval == 0:
            infos['episode_return'] = self.cenv.get_returns()
            infos['sum_rewards'] = self.sum_rewards
            infos['has_key'] = self.cenv.has_key()
            self.sum_rewards = 0

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, infos)
