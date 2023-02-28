# TODO: Separate into mock and non mock tests once we have more mock envs

from pdb import set_trace as T

import random
import numpy as np

import pufferlib
import pufferlib.binding
from environments import bindings

import mock_environments


def test_pack_and_batch_obs():
    for binding in bindings.values():
        env = binding.env_creator()
        obs = env.reset()
        packed = pufferlib.emulation._pack_and_batch_obs(obs)
        assert type(packed) == np.ndarray
        assert len(packed) == len(obs)

def test_flatten():
    inputs = [
        {
            'foo': {
                'bar': 0
            },
            'baz': 1
        },
        0
    ]
    outputs = [
        {
            ('foo', 'bar'): 0,
            ('baz',): 1
        },
        0
    ]
    for inp, out in zip(inputs, outputs):
        test_out = pufferlib.emulation._flatten(inp)
        assert out == test_out, f'\n\tOutput: {test_out}\n\tExpected: {out}'

def test_unflatten():
    input = [1, 2, 3]

    structures = [
        {
            'foo': None,
            'bar': None,
            'baz': None,
        },
        {
            'foo': {
                'bar': None,
                'baz': None,
            },
            'qux': None,
        },
    ]

    outputs = [
        {
            'foo': 1,
            'bar': 2,
            'baz': 3,
        },
        {
            'foo': {
                'bar': 1,
                'baz': 2,
            },
            'qux': 3,
        }
    ]


    for struct, out in zip(structures, outputs):
        test_out = pufferlib.emulation._unflatten(input, struct)
        assert out == test_out, f'\n\tOutput: {test_out}\n\tExpected: {out}'

def test_raw_vs_emulation(binding, seed=42, steps=32):
    raw_env = binding.raw_env_creator()
    raw_env.seed(seed)
    raw_obs = raw_env.reset()

    puf_env = binding.env_creator()
    puf_env.seed(seed)
    puf_obs = puf_env.reset()

    for step in range(steps):
        if binding.emulate_multiagent:
            raw_actions = raw_env.action_space.sample()
        else:
            raw_actions = {a: raw_env.action_space(a).sample() for a in raw_env.agents}

        puf_actions = {a: puf_env.action_space(a).sample() for a in puf_env.possible_agents}

        raw_obs, raw_reward, raw_done, raw_info = raw_env.step(raw_actions)
        puf_obs, puf_reward, puf_done, puf_info = puf_env.step(puf_actions)

        if binding.emulate_multiagent:
            assert np.array_equal(pufferlib.emulation._flatten_ob(raw_obs), puf_obs[1])
            assert raw_reward == puf_reward[1]
            assert raw_done == puf_done[1]
            assert raw_info == puf_info[1]
        else:
            for a in raw_obs:
                assert np.array_equal(pufferlib.emulation._flatten_ob(raw_obs[a]), puf_obs[a]), f'Agent {a} has different observations'
                assert raw_reward[a] == puf_reward[a], f'Agent {a} has different rewards'
                assert raw_done[a] == puf_done[a], f'Agent {a} has different dones'
                assert raw_info[a] == puf_info[a], f'Agent {a} has different infos'

if __name__ == '__main__':
    test_raw_vs_emulation(pufferlib.binding.auto(env_cls=mock_environments.TestEnv))
    #test_raw_vs_emulation(pufferlib.registry.NetHack())
    #test_raw_vs_emulation(pufferlib.registry.Atari('BreakoutNoFrameskip-v4'))

    test_flatten()
    test_unflatten()
    test_pack_and_batch_obs()