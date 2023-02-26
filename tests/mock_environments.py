import numpy as np

import random
import functools

import gym

from pettingzoo.utils.env import ParallelEnv


class TestEnv(ParallelEnv):
    '''
    A complex testing environment with:
        - Multiple and variable agent population
        - Hierarchical observation and action spaces
    '''
    def __init__(self, initial_agents=1, max_agents=100,
                 spawn_attempts_per_tick=2, death_per_tick=1):

        assert death_per_tick <= initial_agents

        self.possible_agents = [abs(hash(str(i))) % 2**16 for i in range(max_agents)]

        self.initial_agents = initial_agents
        self.max_agents = max_agents
        self.spawn_attempts_per_tick = spawn_attempts_per_tick
        self.death_per_tick = death_per_tick

    def reset(self):
        self.tick = 0
        self.agents = random.sample(self.possible_agents, self.initial_agents)
        return {a: self.observation_space(a).sample() for a in self.agents}

    def step(self, action):
        random.seed(self.tick)

        obs, rewards, dones, infos = {}, {}, {}, {}

        dead  = random.sample(self.agents, self.death_per_tick)
        for kill in dead:
            self.agents.remove(kill)
            obs[kill] = self._fill(self.observation_space(kill).sample(), kill)
            rewards[kill] = -1
            dones[kill] = True
            infos[kill] = {'dead': True}

        for spawn in range(self.spawn_attempts_per_tick):
            spawn = random.choice(self.possible_agents)
            if spawn not in self.agents + dead:
                self.agents.append(spawn)
                obs[spawn] = self._fill(self.observation_space(spawn).sample(), spawn)
                rewards[spawn] = 0.1 * random.random()
                dones[spawn] = False
                infos[spawn] = {'dead': False}

        self.tick += 1
        return obs, rewards, dones, infos

    def _fill(self, ob, agent):
        ob['foo'] = np.arange(23, dtype=np.float32) + agent
        ob['bar'] = np.arange(45, dtype=np.float32) + agent
        ob['baz']['qux'] = np.arange(6, dtype=np.float32) + agent
        ob['baz']['quux'] = np.arange(7, dtype=np.float32) + agent
        return ob

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Dict({
            'foo': gym.spaces.Box(low=0, high=1, shape=(23,)),
            'bar': gym.spaces.Box(low=0, high=1, shape=(45,)),
            'baz': gym.spaces.Dict({
                'qux': gym.spaces.Box(low=0, high=1, shape=(6,)),
                'quux': gym.spaces.Box(low=0, high=1, shape=(7,)),
            })
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Dict({
            'foo': gym.spaces.Discrete(2),
            'bar': gym.spaces.Dict({
                'baz': gym.spaces.Discrete(7),
                'qux': gym.spaces.Discrete(8),
            })
        })

    def pack_ob(self, ob):
        # Note: there's currently a weird sort order on obs
        return np.concatenate([ob['bar'], ob['foo'],  ob['baz']['quux'], ob['baz']['qux']])