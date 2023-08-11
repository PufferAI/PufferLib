from pdb import set_trace as T
import numpy as np

import gym
import time

import pufferlib.registry
import pufferlib.utils

import pufferlib.vectorization

from mock_environments import make_performance_env


def _test_vectorization_performance(vectorization, delay, steps, num_workers, envs_per_worker):

    import warnings

    # Convert the specific warning into an error
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Your code here that's triggering the warning

        envs = vectorization(
            env_creator=make_performance_env,
            env_args=[delay, 1],
            num_workers=num_workers,
            envs_per_worker=envs_per_worker,
        )

        envs.reset()
        actions = [[1]] * num_workers * envs_per_worker

        start = time.time()
        for i in range(steps):
            envs.step(actions)
        total_time = time.time() - start

        return steps / total_time


if __name__ == '__main__':
    steps = 1000
    delays = [0, 0.0001, 0.001]
    vectorization = [
        pufferlib.vectorization.Serial,
        pufferlib.vectorization.Multiprocessing,
        #pufferlib.vectorization.Ray,
    ]

    for delay in delays:
        for v in vectorization:
            sps = _test_vectorization_performance(v, delay, steps, 2, 1)
            print(f'{v.__name__}: {steps} steps with {delay} delay: {sps} steps per second')