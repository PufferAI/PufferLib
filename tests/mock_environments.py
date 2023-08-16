from pdb import set_trace as T
import numpy as np

import time

import gym
from pettingzoo.utils.env import ParallelEnv

import pufferlib
import pufferlib.emulation
import pufferlib.utils


HIGH = 100
LOW = 0

def make_performance_env(delay=0, bandwidth=1):
    return pufferlib.emulation.PettingZooPufferEnv(
        env_creator=PerformanceEnv,
        env_args=[delay, bandwidth],
    )

class PerformanceEnv:
    def __init__(self, delay=0, bandwith=1):
        self.agents = [1]
        self.possible_agents = [1]
        self.done = False

        self.delay = delay
        assert bandwith > 0
        self.bandwidth = bandwith

    def reset(self, seed=None):
        return {1: self.observation_space(1).sample()}

    def step(self, actions):
        obs = {1: np.array([0], dtype=np.float32)}
        rewards = {1: 1}
        dones = {1: False}
        infos = {1: {}}

        # Busy wait so process does not swap on sleep
        end = time.perf_counter() + self.delay
        while time.perf_counter() < end:
            pass

        return obs, rewards, dones, infos

    def observation_space(self, agent):
        return gym.spaces.Box(
            low=-2**20, high=2**20,
            shape=(self.bandwidth,), dtype=np.float32
        )

    def action_space(self, agent):
        return gym.spaces.Discrete(2)
    

### Other Mock environments and utilities
def _agent_str_to_int(agent):
    return int(agent.split('_')[-1])

def _sample_space(agent, tick, space, zero=False):
    if type(agent) is str:
        agent = float(agent.split('_')[-1])

    if isinstance(space, gym.spaces.Discrete):
        if zero:
            return 0
        return hash(f'{agent}-{tick}') % space.n
    elif isinstance(space, gym.spaces.Box):
        if zero:
            return np.zeros(space.shape, dtype=np.float32)
        nonce = (agent % HIGH + tick/10) % (HIGH - 1)
        return np.array([nonce+0.01*t for t in range(space.shape[0])], dtype=np.float32)
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(_sample_space(agent, tick, s) for s in space.spaces)
    elif isinstance(space, gym.spaces.Dict):
        return {k: _sample_space(agent, tick, v) for k, v in space.spaces.items()}
    else:
        raise ValueError(f"Invalid space type: {type(space)}")

def make_mock_singleagent_env(observation_space, action_space):
    class TestEnv(gym.Env):
        def __init__(self):
            self.observation_space = observation_space
            self.action_space = action_space

        def reset(self, seed=None):
            self.tick = 0
            self.rng = pufferlib.utils.RandomState(seed)

            return _sample_space('agent_1', self.tick, observation_space)

        def step(self, actions):
            reward = self.tick
            done = self.tick < 10
            self.tick += 1

            return (
                _sample_space('agent_1', self.tick, observation_space),
                reward, done, {'dead': done})

    return TestEnv


def make_mock_multiagent_env(
        observation_space,
        action_space,
        initial_agents,
        max_agents,
        spawn_per_tick,
        death_per_tick,
        homogeneous_spaces=True):
    class TestEnv(ParallelEnv):
        def __init__(self):
            self.possible_agents = [f'agent_{i+1}' for i in range(max_agents)]
            self.agents = []

        def reset(self, seed=None):
            self.tick = 0
            self.agents = self.possible_agents[:initial_agents]

            return {a: _sample_space(a, self.tick, observation_space)
                for a in self.agents}

        def step(self, actions):
            obs, rewards, dones, infos = {}, {}, {}, {}
            self.tick += 1

            dead  = self.agents[:death_per_tick]
            for kill in dead:
                self.agents.remove(kill)
                # TODO: Make pufferlib work without pad obs
                # but still require rewards, dones, and optionally infos
                obs[kill] = _sample_space(kill, self.tick, observation_space, zero=True)
                rewards[kill] = -1
                dones[kill] = True
                infos[kill] = {'dead': True}

            # TODO: Fix this
            assert spawn_per_tick == 0
            for spawn in range(spawn_per_tick):
                # TODO: Make pufferlib check if an agent respawns on the
                # Same tick as it dies (is this good or bad?)
                spawn = self.rng.choice(self.possible_agents)
                if spawn not in self.agents + dead:
                    self.agents.append(spawn)

            for agent in self.agents:
                obs[agent] = _sample_space(agent, self.tick, observation_space)
                rewards[agent] = 0.1 * _agent_str_to_int(agent)
                dones[agent] = False
                infos[agent] = {'dead': False}

            return obs, rewards, dones, infos

        def observation_space(self, agent) -> gym.Space:
            return observation_space

        def action_space(self, agent) -> gym.Space:
            return action_space

        def render(self, mode='human'):
            pass

        def close(self):
            pass

    return TestEnv


MOCK_OBSERVATION_SPACES = [
    # Simple spaces
    gym.spaces.Box(low=LOW, high=HIGH, shape=(4,), dtype=np.float32),
    #gym.spaces.Discrete(5),

    # Nested spaces
    gym.spaces.Dict({
        "foo": gym.spaces.Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
        "bar": gym.spaces.Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
    }),
    #gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(4))),
    gym.spaces.Tuple((
        gym.spaces.Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
        #gym.spaces.Discrete(3),
        gym.spaces.Dict({
            "baz": gym.spaces.Box(low=LOW, high=HIGH, shape=(1,), dtype=np.float32),
            "qux": gym.spaces.Box(low=LOW, high=HIGH, shape=(1,), dtype=np.float32),
        }),
    )),
    gym.spaces.Dict({
        "foo": gym.spaces.Tuple((
            gym.spaces.Box(low=LOW, high=HIGH, shape=(2,), dtype=np.float32),
            #gym.spaces.Discrete(3),
        )),
        #"bar": gym.spaces.Dict({
        #    "baz": gym.spaces.Discrete(2),
        #    "qux": gym.spaces.Discrete(4),
        #}),
    }),
]


MOCK_ACTION_SPACES = [
    # Simple spaces
    gym.spaces.Discrete(5),

    # Nested spaces
    #gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3))),
    gym.spaces.Dict({
        "foo": gym.spaces.Discrete(4),
        "bar": gym.spaces.Discrete(2),
    }),
    # gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    #gym.spaces.Tuple((
    #    gym.spaces.Discrete(4),
    #    gym.spaces.Dict({
    #        "baz": gym.spaces.Discrete(2),
    #        "qux": gym.spaces.Discrete(2),
    #    }),
    #)),
    # gym.spaces.Dict({
    #     "foo": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
    #     "bar": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
    # }),
    gym.spaces.Dict({
        "foo": gym.spaces.Tuple((
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(3),
        )),
        "bar": gym.spaces.Dict({
            "baz": gym.spaces.Discrete(2),
            "qux": gym.spaces.Discrete(4),
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

MOCK_SINGLE_AGENT_ENVIRONMENTS = []
MOCK_MULTI_AGENT_ENVIRONMENTS = []
for obs_space in MOCK_OBSERVATION_SPACES:
    for act_space in MOCK_ACTION_SPACES:
        MOCK_SINGLE_AGENT_ENVIRONMENTS.append(
            make_mock_singleagent_env(
                observation_space=obs_space,
                action_space=act_space,
            )
        )
 
        MOCK_MULTI_AGENT_ENVIRONMENTS.append(
            make_mock_multiagent_env(
                observation_space=obs_space,
                action_space=act_space,
                initial_agents=16,
                max_agents=16,
                spawn_per_tick=0,
                death_per_tick=1,
            )
        )