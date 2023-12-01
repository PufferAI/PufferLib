from pdb import set_trace as T
import numpy as np

import time
import hashlib

import gym
import gymnasium
from gymnasium.spaces import Box, Discrete, Dict, Tuple
from pufferlib import spaces
from pettingzoo.utils.env import ParallelEnv

import pufferlib
import pufferlib.emulation
import pufferlib.utils


HIGH = 100
LOW = 0

MOCK_OBSERVATION_SPACES = [
    # Atari space
    Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),

    # NetHack space
    Dict({
        'blstats': Box(-2147483648, 2147483647, (27,), 'int64'),
        'chars': Box(0, 255, (21, 79), 'uint8'),
        'colors': Box(0, 15, (21, 79), 'uint8'),
        'glyphs': Box(0, 5976, (21, 79), 'int16'),
        'inv_glyphs': Box(0, 5976, (55,), 'int16'),
        'inv_letters': Box(0, 127, (55,), 'uint8'),
        'inv_oclasses': Box(0, 18, (55,), 'uint8'),
        'inv_strs': Box(0, 255, (55, 80), 'uint8'),
        'message': Box(0, 255, (256,), 'uint8'),
        'screen_descriptions': Box(0, 127, (21, 79, 80), 'uint8'),
        'specials': Box(0, 255, (21, 79), 'uint8'),
        'tty_chars': Box(0, 255, (24, 80), 'uint8'),
        'tty_colors': Box(0, 31, (24, 80), 'int8'),
        'tty_cursor': Box(0, 255, (2,), 'uint8'),
    }),
    
    # Neural MMO space
    Dict({
        'ActionTargets': Dict({
            'Attack': Dict({
                'Style': Box(0, 1, (3,), 'int8'),
                'Target': Box(0, 1, (100,), 'int8'),
            }),
            'Buy': Dict({
                'MarketItem': Box(0, 1, (1024,), 'int8'),
            }),
            'Comm': Dict({
                'Token': Box(0, 1, (50,), 'int8'),
            }),
            'Destroy': Dict({
                'InventoryItem': Box(0, 1, (12,), 'int8'),
            }),
            'Give': Dict({
                'InventoryItem': Box(0, 1, (12,), 'int8'),
                'Target': Box(0, 1, (100,), 'int8'),
            }),
            'GiveGold': Dict({
                'Price': Box(0, 1, (99,), 'int8'),
                'Target': Box(0, 1, (100,), 'int8'),
            }),
            'Move': Dict({
                'Direction': Box(0, 1, (5,), 'int8'),
            }),
            'Sell': Dict({
                'InventoryItem': Box(0, 1, (12,), 'int8'),
                'Price': Box(0, 1, (99,), 'int8'),
            }),
            'Use': Dict({
                'InventoryItem': Box(0, 1, (12,), 'int8'),
            })
        }),
        'AgentId': Discrete(129),
        'CurrentTick': Discrete(1025),
        'Entity': Box(-32768, 32767, (100, 23), 'int16'),
        'Inventory': Box(-32768, 32767, (12, 16), 'int16'),
        'Market': Box(-32768, 32767, (1024, 16), 'int16'),
        'Task': Box(-32770.0, 32770.0, (1024,), 'float16'),
        'Tile': Box(-32768, 32767, (225, 3), 'int16'),
    }),

    # Simple spaces
    Discrete(5),
    Box(low=LOW, high=HIGH, shape=(4,), dtype=np.float32),

    # Nested spaces
    Dict({
        "foo": Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
        "bar": Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
    }),
    Tuple((Discrete(3), Discrete(4))),
    Tuple((
        Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
        Discrete(3),
        Dict({
            "baz": Box(low=LOW, high=HIGH, shape=(1,), dtype=np.float32),
            "qux": Box(low=LOW, high=HIGH, shape=(1,), dtype=np.float32),
        }),
    )),
    Dict({
        "foo": Tuple((
            Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
            Discrete(3),
        )),
        "bar": Dict({
            "baz": Discrete(2),
            "qux": Discrete(4),
        }),
    }),
]

MOCK_ACTION_SPACES = [
    # NetHack action space
    Discrete(5),

    # Neural MMO action space
    Dict({
        'Attack': Dict({
            'Style': Discrete(3),
            'Target': Discrete(100),
        }),
        'Buy': Dict({
            'MarketItem': Discrete(1024),
        }),
        'Comm': Dict({
            'Token': Discrete(50),
        }),
        'Destroy': Dict({
            'InventoryItem': Discrete(12),
        }),
        'Give': Dict({
            'InventoryItem': Discrete(12),
            'Target': Discrete(100),
        }),
        'GiveGold': Dict({
            'Price': Discrete(99),
            'Target': Discrete(100),
        }),
        'Move': Dict({
            'Direction': Discrete(5),
        }),
        'Sell': Dict({
            'InventoryItem': Discrete(12),
            'Price': Discrete(99),
        }),
        'Use': Dict({
            'InventoryItem': Discrete(12),
        })
    }),

    # Nested spaces
    Tuple((Discrete(2), Discrete(3))),
    Dict({
        "foo": Discrete(4),
        "bar": Discrete(2),
    }),
    Tuple((
        Discrete(4),
        Dict({
            "baz": Discrete(2),
            "qux": Discrete(2),
        }),
    )),
    Dict({
        "foo": Tuple((
            Discrete(2),
            Discrete(3),
        )),
        "bar": Dict({
            "baz": Discrete(2),
            "qux": Discrete(4),
        }),
    }),
]

MOCK_TEAMS = {
    'None': None,
    'single': {
        'team_1': ['agent_1'],
        'team_2': ['agent_2'],
        'team_3': ['agent_3'],
        'team_4': ['agent_4'],
        'team_5': ['agent_5'],
        'team_6': ['agent_6'],
        'team_7': ['agent_7'],
        'team_8': ['agent_8'],
        'team_9': ['agent_9'],
        'team_10': ['agent_10'],
        'team_11': ['agent_11'],
        'team_12': ['agent_12'],
        'team_13': ['agent_13'],
        'team_14': ['agent_14'],
        'team_15': ['agent_15'],
        'team_16': ['agent_16'],
    },
    'pairs': {
        'team_1': ['agent_1', 'agent_2'],
        'team_2': ['agent_3', 'agent_4'],
        'team_3': ['agent_5', 'agent_6'],
        'team_4': ['agent_7', 'agent_8'],
        'team_5': ['agent_9', 'agent_10'],
        'team_6': ['agent_11', 'agent_12'],
        'team_7': ['agent_13', 'agent_14'],
        'team_8': ['agent_15', 'agent_16'],
    },
    'mixed': {
        'team_1': ['agent_1', 'agent_2'],
        'team_2': ['agent_3', 'agent_4', 'agent_5', 'agent_6'],
        'team_3': ['agent_7', 'agent_8', 'agent_9'],
        'team_4': ['agent_10', 'agent_11', 'agent_12', 'agent_13', 'agent_14'],
        'team_5': ['agent_15', 'agent_16'],
    },
}

DEFAULT_OBSERVATION_SPACE = gymnasium.spaces.Box(
    low=-2**20, high=2**20,
    shape=(1,), dtype=np.float32
)
DEFAULT_ACTION_SPACE = gymnasium.spaces.Discrete(2)
 

def make_all_mock_environments():
    mock_single_agent_environments = []
    mock_multi_agent_environments = []
    for obs_space in MOCK_OBSERVATION_SPACES:
        for act_space in MOCK_ACTION_SPACES:
            mock_single_agent_environments.append(
                GymnasiumTestEnv(
                    observation_space=obs_space,
                    action_space=act_space,
                )
            )
     
            mock_multi_agent_environments.append(
                PettingZooTestEnv(
                    observation_space=obs_space,
                    action_space=act_space,
                    initial_agents=16,
                    max_agents=16,
                    spawn_per_tick=0,
                    death_per_tick=1,
                )
            )
    return mock_single_agent_environments, mock_multi_agent_environments

def do_work(delay_mean, delay_std):
    start, idx = time.process_time(), 0
    target_time = delay_mean + delay_std*np.random.randn()
    while time.process_time() - start < target_time:
        idx += 1
    return

class GymnasiumPerformanceEnv:
    def __init__(self, delay_mean=0, delay_std=0):
        self.observation_space = DEFAULT_OBSERVATION_SPACE
        self.action_space = DEFAULT_ACTION_SPACE
        self.observation = self.observation_space.sample()

        self.delay_mean = delay_mean
        self.delay_std = delay_std

        # Test performance independent of PufferLib seeding
        np.random.seed(time.time_ns() % 2**32)

    def reset(self, seed=None):
        return self.observation, {}

    def step(self, action):
        do_work(self.delay_mean, self.delay_std)
        return self.observation, 0, False, False, {}

    def close(self):
        pass

class PettingZooPerformanceEnv:
    def __init__(self, delay_mean, delay_std):
        self.possible_agents = [1]
        self.agents = [1]
        self.done = False

        self.delay_mean = delay_mean
        self.delay_std = delay_std

    def observation_space(self, agent):
        return DEFAULT_OBSERVATION_SPACE

    def action_space(self, agent):
        return DEFAULT_ACTION_SPACE
 
    def reset(self, seed=None):
        return {1: self.observation_space(1).sample()}, {1: {}}

    def step(self, actions):
        obs = {1: np.array([0], dtype=np.float32)}
        rewards = {1: 1}
        dones = {1: False}
        truncateds = {1: False}
        infos = {1: {}}

        do_work(self.delay_mean, self.delay_std)

        return obs, rewards, dones, truncateds, infos

    def close(self):
        pass

class GymnasiumTestEnv(gym.Env):
    def __init__(self,
            observation_space=DEFAULT_OBSERVATION_SPACE,
            action_space=DEFAULT_ACTION_SPACE):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, seed=None):
        self.tick = 0
        self.rng = pufferlib.utils.RandomState(seed)

        ob = _sample_space('agent_1', self.tick, self.observation_space)
        return ob, {}

    def step(self, actions):
        reward = self.tick
        done = self.tick < 10
        self.tick += 1

        ob = _sample_space('agent_1', self.tick, self.observation_space)
        return ob, reward, done, False, {'dead': done}

    def close(self):
        pass

class PettingZooTestEnv(ParallelEnv):
    def __init__(self,
            observation_space=DEFAULT_OBSERVATION_SPACE,
            action_space=DEFAULT_ACTION_SPACE,
            initial_agents=16, max_agents=16,
            spawn_per_tick=0, death_per_tick=1,
            homogeneous_spaces=True):
        self._observation_space = observation_space
        self._action_space = action_space
        self.initial_agents = initial_agents
        self.max_agents = max_agents
        self.spawn_per_tick = spawn_per_tick
        self.death_per_tick = death_per_tick
        self.homogeneous_spaces = homogeneous_spaces

        self.possible_agents = [f'agent_{i+1}' for i in range(max_agents)]
        self.agents = []

    def reset(self, seed=None):
        self.tick = 0
        self.agents = self.possible_agents[:self.initial_agents]

        obs = {a: _sample_space(a, self.tick, self._observation_space)
            for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        obs, rewards, dones, truncateds, infos = {}, {}, {}, {}, {}
        self.tick += 1

        dead  = self.agents[:self.death_per_tick]
        for kill in dead:
            self.agents.remove(kill)
            # TODO: Make pufferlib work without pad obs
            # but still require rewards, dones, and optionally infos
            obs[kill] = _sample_space(kill, self.tick,
                self._observation_space, zero=True)
            rewards[kill] = -1
            dones[kill] = True
            truncateds[kill] = False
            infos[kill] = {'dead': True}

        # TODO: Fix this
        assert self.spawn_per_tick == 0
        for spawn in range(self.spawn_per_tick):
            # TODO: Make pufferlib check if an agent respawns on the
            # Same tick as it dies (is this good or bad?)
            spawn = self.rng.choice(self.possible_agents)
            if spawn not in self.agents + dead:
                self.agents.append(spawn)

        for agent in self.agents:
            obs[agent] = _sample_space(agent, self.tick, self._observation_space)
            rewards[agent] = 0.1 * _agent_str_to_int(agent)
            dones[agent] = False
            truncateds[agent] = False
            infos[agent] = {'dead': False}

        return obs, rewards, dones, truncateds, infos

    def observation_space(self, agent) -> gym.Space:
        return self._observation_space

    def action_space(self, agent) -> gym.Space:
        return self._action_space

    def render(self, mode='human'):
        pass

    def close(self):
        pass

### Other Mock environments and utilities
def _agent_str_to_int(agent):
    return int(agent.split('_')[-1])

def _sample_space(agent, tick, space, zero=False):
    if isinstance(agent, str):
        agent = float(agent.split('_')[-1])

    if isinstance(space, spaces.Discrete):
        if zero:
            return 0
        return int((10*agent + tick) % space.n)
    elif isinstance(space, spaces.Box):
        if zero:
            return np.zeros(space.shape, dtype=space.dtype)

        # Try to make a relatively unique data pattern
        # without using RNG
        nonce = 10*agent + tick
        low = space.low
        high = space.high
        sample = low + np.arange(low.size).reshape(space.shape) + nonce
        sample = (sample % high).astype(space.dtype)
        return sample
    elif isinstance(space, spaces.Tuple):
        return tuple(_sample_space(agent, tick, s, zero) for s in space.spaces)
    elif isinstance(space, spaces.Dict):
        return {k: _sample_space(agent, tick, v, zero) for k, v in space.spaces.items()}
    else:
        raise ValueError(f"Invalid space type: {type(space)}")
