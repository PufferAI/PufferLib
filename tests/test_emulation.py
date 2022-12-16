from pdb import set_trace as T
import numpy as np

import pufferlib
from environments import bindings

def test_pack_and_batch_obs():
    for binding in bindings:
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
    test_pack_and_batch_obs()