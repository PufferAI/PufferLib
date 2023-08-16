from pdb import set_trace as T

import numpy as np
import timeit
from collections import OrderedDict

import gym

from pufferlib.emulation import flatten_space, flatten, unflatten, concatenate, split, python_flatten, python_unflatten
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

def test_python_flatten_unflatten():
    for space in nested_spaces:
        sample = space.sample()
        flat_sample = python_flatten(sample)
        unflat_sample = python_unflatten(flat_sample, space)

        assert pufferlib.utils._compare_space_samples(sample, unflat_sample), "Unflatten failed."

def test_flatten_unflatten():
    for space in nested_spaces:
        sample = space.sample()
        flat_sample = flatten(sample)
        unflat_sample = unflatten(flat_sample)

        assert pufferlib.utils._compare_space_samples(sample, unflat_sample), "Unflatten failed."

def test_python_flatten_implementation_speed():
    for space in nested_spaces:
        sample = space.sample()
        flat_sample = python_flatten(sample)
        flatten_time = timeit.timeit(lambda: python_flatten(sample), number=1000)
        unflatten_time = timeit.timeit(lambda: python_unflatten(flat_sample, space), number=1000)
        print(f"Space: {space}\n\tFlatten time: {flatten_time}, Unflatten time: {unflatten_time}")

def test_flatten_implementation_speed():
    for space in nested_spaces:
        sample = space.sample()
        flat_sample = flatten(sample)
        flatten_time = timeit.timeit(lambda: flatten(sample), number=1000)
        unflatten_time = timeit.timeit(lambda: unflatten(flat_sample), number=1000)
        print(f"Space: {space}\n\tFlatten time: {flatten_time}, Unflatten time: {unflatten_time}")

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

    test_python_flatten_unflatten()
    test_flatten_unflatten()

    test_python_flatten_implementation_speed()
    test_flatten_implementation_speed()
    #test_pack_unpack()