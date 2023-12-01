from pdb import set_trace as T

import numpy as np
import timeit
import gym

from pufferlib.emulation import flatten_structure, flatten_space, flatten, unflatten, concatenate, split
import pufferlib.utils

def test_pack_unpack():
    for space in nested_spaces:
        sample = space.sample()
        flat_space = flatten_space(space)
        flat_sample = flatten(sample)
        pack_sample = concatenate(flat_sample)
        unpack_sample = split(pack_sample, flat_space, batched=False)
        unflat_sample = unflatten(unpack_sample, space)
        assert pufferlib.utils.compare_space_samples(sample, unflat_sample), "Unflatten failed."
 
test_cases = [
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


def test_flatten_unflatten(iterations=10_000):
    flatten_times = []
    concatenate_times = []
    split_times = []
    unflatten_times = []
    for space in test_cases:
        data = space.sample()
        flat = flatten(data)
        structure = flatten_structure(data)
        flat_space = flatten_space(space)
        merged = concatenate(flat)
        unmerged = split(merged, flat_space, batched=False)
        unflat = unflatten(unmerged, structure)
        assert pufferlib.utils.compare_space_samples(data, unflat), "Unflatten failed."

        flatten_times.append(timeit.timeit(
            lambda: flatten(data), number=iterations))
        concatenate_times.append(timeit.timeit(
            lambda: concatenate(flat), number=iterations))
        split_times.append(timeit.timeit(
            lambda: split(merged, flat_space, batched=False), number=iterations))
        unflatten_times.append(timeit.timeit(
            lambda: unflatten(unmerged, structure), number=iterations))

    print(f'{np.mean(flatten_times)/iterations:.8f}: Flatten time')
    print(f'{np.mean(concatenate_times)/iterations:.8f}: Concatenate time')
    print(f'{np.mean(split_times)/iterations:.8f}: Split time')
    print(f'{np.mean(unflatten_times)/iterations:.8f}: Unflatten time')


if __name__ == '__main__':
    iterations = 10_000
    test_flatten_unflatten(iterations=iterations)
