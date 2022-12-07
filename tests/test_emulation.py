from pdb import set_trace as T
import numpy as np

#from pettingzoo.butterfly import knights_archers_zombies_v10
#from pettingzoo.utils.conversions import aec_to_parallel

import nmmo

import pufferlib


def setup():
    #kaz = aec_to_parallel(knights_archers_zombies_v10.env(max_zombies=0))
    mmo = nmmo.Env()

    return [mmo]


def test_pack_obs_space():
    for env in setup():
        obs = env.reset()
        obs_space = env.observation_space(0)
        packed = pufferlib.emulation.pack_obs_space(obs_space)

        ob = list(obs.values())[0]
        flat = pufferlib.emulation.flatten_ob(ob)
        assert packed.shape == flat.shape

def test_pack_atn_space():
    for env in setup():
        obs = env.reset()
        atn_space = env.action_space(0)
        packed = pufferlib.emulation.pack_atn_space(atn_space)

def test_pack_and_batch_obs():
    for env in setup():
        obs = env.reset()
        packed = pufferlib.emulation.pack_and_batch_obs(obs)
        assert type(packed) == np.ndarray
        assert len(packed) == len(obs)

def test_unpack_batched_obs():
    for env in setup():
        obs = env.reset()
        packed = pufferlib.emulation.pack_and_batch_obs(obs)

        obs_space = env.observation_space(0)
        unpacked = pufferlib.emulation.unpack_batched_obs(obs_space, packed)

        assert obs[1].keys() == unpacked.keys()

def test_unflatten_atn():
    for env in setup():
        atn = env.action_space(1).sample()
        atn = pufferlib.emulation.pack_atn_space(atn)
        atn = [e for e in atn.values()]
        #atn = {1: atn}
        recovered = pufferlib.emulation.unflatten(atn, env.action_space(1))
        obs = env.reset()
        for k, orig in obs.items():
            flat = pufferlib.emulation.flatten(orig)
            recovered = pufferlib.emulation.unflatten(flat, env.observation_space(k))
            assert flat == recovered

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
        }
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

if __name__ == '__main__':
    test_flatten()
    test_unflatten()
    test_unflatten_atn()
    test_unpack_batched_obs()
    test_pack_and_batch_obs()
    test_pack_obs_space()
    test_pack_atn_space()