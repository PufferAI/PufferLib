from pdb import set_trace as T
import numpy as np

import gym
import time

import pufferlib.registry
import pufferlib.utils

import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.multiprocessing_shared
import pufferlib.vectorization.ray
import pufferlib.vectorization.ray_shared
import pufferlib.vectorization.serial

from mock_environments import PerformanceBinding


def _test_vectorization_performance(binding, vec_backend, steps, num_workers, envs_per_worker):
    actions = [1] * num_workers * envs_per_worker

    with pufferlib.utils.Suppress():
        envs = vec_backend(
            binding,
            num_workers=num_workers,
            envs_per_worker=envs_per_worker,
        )
        envs.reset()

    start = time.time()
    for i in range(steps):
        envs.step(actions)
    total_time = time.time() - start

    envs.close()
    return steps / total_time


if __name__ == '__main__':
    steps = 1000
    delays = [0, 0.0001, 0.001, 0.01]
    backends = [
        pufferlib.vectorization.serial.VecEnv,
        pufferlib.vectorization.multiprocessing.VecEnv,
        pufferlib.vectorization.multiprocessing_shared.VecEnv,
        pufferlib.vectorization.ray.VecEnv,
        pufferlib.vectorization.ray_shared.VecEnv,
    ]

    for d in delays:
        for b in backends:
            binding = PerformanceBinding(delay=d, bandwith=100_000)
            sps = _test_vectorization_performance(binding, b, steps, 2, 1)
            print(f'{b.__module__.split(".")[-1]}: {steps} steps with {d} delay: {sps} steps per second')

    #import pufferlib.registry.atari
    #binding = pufferlib.registry.atari.make_binding('BreakoutNoFrameskip-v4', framestack=1)