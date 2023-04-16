from pdb import set_trace as T
import numpy as np

import functools

import gym
from pettingzoo.utils.env import ParallelEnv

import pufferlib
import pufferlib.utils

def _agent_str_to_int(agent):
    return int(agent.split('_')[-1])

def _sample_space(agent, tick, space):
    if isinstance(space, gym.spaces.Discrete):
        return hash(f'{agent}-{tick}') % space.n
    elif isinstance(space, gym.spaces.Box):
        return np.linspace(space.low, space.high, num=space.shape[0]) * (tick % 2)
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
            reward = 0.1 * self.rng.random()
            done = reward < 0.01
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

        def reset(self, seed=None):
            self.tick = 0
            self.agents = self.possible_agents[:initial_agents]

            return {a: self._sample_space(a, self.tick, observation_space)
                for a in self.agents}

        def step(self, actions):
            obs, rewards, dones, infos = {}, {}, {}, {}

            dead  = self.agents[:death_per_tick]
            for kill in dead:
                self.agents.remove(kill)
                # TODO: Make pufferlib work without pad obs
                # but still require rewards, dones, and optionally infos
                obs[kill] = self._sample_space(kill, self.tick, observation_space)
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
                obs[agent] = self._sample_space(agent, self.tick, observation_space)
                rewards[agent] = 0.1 * _agent_str_to_int(agent)
                dones[agent] = False
                infos[agent] = {'dead': False}

            self.tick += 1
            return obs, rewards, dones, infos

        def _sample_space(self, agent, tick, space):
            if isinstance(space, gym.spaces.Discrete):
                return hash(f'{agent}-{tick}') % space.n
            elif isinstance(space, gym.spaces.Box):
                return np.linspace(space.low, space.high, num=space.shape[0]) * (tick % 2)
            elif isinstance(space, gym.spaces.Tuple):
                return tuple(self._sample_space(agent, tick, s) for s in space.spaces)
            elif isinstance(space, gym.spaces.Dict):
                return {k: self._sample_space(agent, tick, v) for k, v in space.spaces.items()}
            else:
                raise ValueError(f"Invalid space type: {type(space)}")

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
    gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    #gym.spaces.Discrete(5),

    # Nested spaces
    gym.spaces.Dict({
        "foo": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        "bar": gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    }),
    #gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(4))),
    gym.spaces.Tuple((
        gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        #gym.spaces.Discrete(3),
        gym.spaces.Dict({
            "baz": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "qux": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        }),
    )),
    gym.spaces.Dict({
        "foo": gym.spaces.Tuple((
            gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
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
    gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3))),
    gym.spaces.Dict({
        "foo": gym.spaces.Discrete(4),
        "bar": gym.spaces.Discrete(2),
    }),
    # gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    gym.spaces.Tuple((
        gym.spaces.Discrete(4),
        gym.spaces.Dict({
            "baz": gym.spaces.Discrete(2),
            "qux": gym.spaces.Discrete(2),
        }),
    )),
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

MOCK_ENVIRONMENTS = []
for obs_space in MOCK_OBSERVATION_SPACES:
    for act_space in MOCK_ACTION_SPACES:
        MOCK_ENVIRONMENTS.append(
            make_mock_multiagent_env(
                observation_space=obs_space,
                action_space=act_space,
                initial_agents=16,
                max_agents=16,
                spawn_per_tick=0,
                death_per_tick=1,
            )
        )