from pdb import set_trace as T
import numpy as np

import gym
import time

import pufferlib.registry
import pufferlib.utils

import pufferlib.vectorization

from mock_environments import PerformanceBinding


def _test_vectorization_performance(binding, vec_backend, steps, num_workers, envs_per_worker):
    actions = [1] * num_workers * envs_per_worker

    #with pufferlib.utils.Suppress():
    envs = pufferlib.vectorization.VecEnv(
        binding,
        vec_backend,
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
    )
    envs.reset()

    # Warmup
    for i in range(10):
        envs.step(actions)

    start = time.perf_counter()
    for i in range(steps):
        envs.step(actions)
    total_time = time.perf_counter() - start

    envs.close()
    return steps * num_workers * envs_per_worker / total_time


if __name__ == '__main__':
    steps = 10000
    delays = [0, 0.0001, 0.001]
    backends = [
        #pufferlib.vectorization.Serial,
        pufferlib.vectorization.Multiprocessing,
    ]

    for d in delays:
        for b in backends:
            binding = PerformanceBinding(delay=d, bandwith=1)
            sps = _test_vectorization_performance(binding, b, steps, 1, 1)
            print(f'{b.__name__.split(".")[-1]}: {steps} steps with {d} delay: {sps} steps per second')

    #import pufferlib.registry.atari
    #binding = pufferlib.registry.atari.make_binding('BreakoutNoFrameskip-v4', framestack=1)