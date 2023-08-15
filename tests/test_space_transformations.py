from pdb import set_trace as T

import numpy as np
import timeit
from collections import OrderedDict

import gym

import pufferlib.emulation

import mock_environments


nested_spaces = [
    # Nested Dict with Box and Discrete spaces
    gym.spaces.Dict({
        "a": gym.spaces.Box(low=0, high=1, shape=(3,)),
        "b": gym.spaces.MultiDiscrete([3, 10]),
        "c": gym.spaces.Dict({
            "d": gym.spaces.Box(low=-10, high=10, shape=(100,)),
            "e": gym.spaces.Discrete(1000)
        })
    }),

    # Nested Tuple with Box spaces of different shapes
    gym.spaces.Tuple((
        gym.spaces.Box(low=0, high=1, shape=(1,)),
        gym.spaces.Box(low=-5, high=5, shape=(10,)),
        gym.spaces.Tuple((
            gym.spaces.Box(low=-100, high=100, shape=(1000,)),
            gym.spaces.Box(low=-1000, high=1000, shape=(10000,))
        ))
    )),

    # Nested Dict with Tuple, Box, and Discrete spaces
    gym.spaces.Dict({
        "f": gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(3))),
        "g": gym.spaces.Box(low=-10, high=10, shape=(50,)),
        "h": gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=1, shape=(500,)),
            gym.spaces.Dict({
                "i": gym.spaces.Discrete(5000),
                "j": gym.spaces.Box(low=-100, high=100, shape=(10000,))
            })
        ))
    }),

    # Flat spaces for control
    gym.spaces.Box(low=0, high=1, shape=(10,)),
    gym.spaces.Discrete(100)
]

def flatten_space(space):
    def _recursion_helper(current):
        if isinstance(current, gym.spaces.Tuple):
            for elem in current:
                _recursion_helper(elem)
        elif isinstance(current, gym.spaces.Dict):
            for value in current.values():
                _recursion_helper(value)
        else:
            flat.append(current)

    flat = []
    _recursion_helper(space)
    return flat

def flatten(sample):
    def _recursion_helper(current):
        if isinstance(current, tuple):
            for elem in current:
                _recursion_helper(elem)
        elif isinstance(current, OrderedDict):
            for value in current.values():
                _recursion_helper(value)
        elif isinstance(current, np.ndarray):
            flat.append(current)
        else:
            flat.append(np.array([current]))

    flat = []
    _recursion_helper(sample)
    return flat

def unflatten(flat_sample, space):
    idx = [0]  # Wrapping the index in a list to maintain the reference
    def _recursion_helper(space):
        if isinstance(space, gym.spaces.Tuple):
            unflattened_tuple = tuple(_recursion_helper(subspace) for subspace in space)
            return unflattened_tuple
        if isinstance(space, gym.spaces.Dict):
            unflattened_dict = OrderedDict((key, _recursion_helper(subspace)) for key, subspace in space.items())
            return unflattened_dict
        if isinstance(space, gym.spaces.Discrete):
            idx[0] += 1
            return int(flat_sample[idx[0] - 1])

        idx[0] += 1
        return flat_sample[idx[0] - 1]

    return _recursion_helper(space)

def concatenate(flat_sample):
    if len(flat_sample) == 1:
        return flat_sample[0]
    return np.concatenate([e.ravel() for e in flat_sample])

def split(stacked_sample, flat_space, batched=True):
    assert isinstance(stacked_sample, np.ndarray), "Input must be a numpy array."

    if batched:
        batch = stacked_sample.shape[0]

    leaves = []
    ptr = 0
    for subspace in flat_space:
        shape = subspace.shape
        typ = subspace.dtype
        sz = int(np.prod(shape))

        if shape == ():
            shape = (1,)

        if batched:
            samp = stacked_sample[:, ptr:ptr+sz].reshape(batch, *shape).astype(typ)
        else:
            samp = stacked_sample[ptr:ptr+sz].reshape(*shape).astype(typ)
            if isinstance(subspace, gym.spaces.Discrete):
                samp = int(samp[0])

        leaves.append(samp)
        ptr += sz

    return leaves

def test_flatten_unflatten():
    for space in nested_spaces:
        sample = space.sample()
        flat_sample = flatten(sample)
        unflat_sample = unflatten(flat_sample, space)

        assert pufferlib.utils._compare_space_samples(sample, unflat_sample), "Unflatten failed."

def test_pack_unpack():
    for space in nested_spaces:
        sample = space.sample()
        flat_space = flatten_space(space)
        flat_sample = flatten(sample)
        pack_sample = concatenate(flat_sample)
        unpack_sample = split(pack_sample, flat_space, batched=False)
        unflat_sample = unflatten(unpack_sample, space)
        assert pufferlib.utils._compare_space_samples(sample, unflat_sample), "Unflatten failed."
 
if __name__ == '__main__':
    # Benchmarking different spaces
    '''
    for space in nested_spaces:
        sample = space.sample()
        flat = flatten(sample)
        flatten_time = timeit.timeit(lambda: flatten(sample), number=1000)
        unflatten_time = timeit.timeit(lambda: unflatten(flat, space), number=1000)
        print(f"Space: {space}\n\tFlatten time: {flatten_time}, Unflatten time: {unflatten_time}")
    '''

    test_flatten_unflatten()
    test_pack_unpack()