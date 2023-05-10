from pdb import set_trace as T
import numpy as np

import gym
import time

import pufferlib.registry
import pufferlib.registry.nmmo

import pufferlib.utils

import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.multiprocessing_shared
import pufferlib.vectorization.ray
import pufferlib.vectorization.ray_shared
import pufferlib.vectorization.serial

import nmmo


def _test_vectorization_performance(binding, vec_backend, steps, num_workers, envs_per_worker):
    actions = np.stack([binding.single_action_space.sample()
            for _ in range(128 * num_workers * envs_per_worker)])

    #with pufferlib.utils.Suppress():
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

    all_profiles = envs.profile()
    envs.close()

    profile_dict = pufferlib.utils.aggregate_profilers(all_profiles)
    total_agent_steps = sum(p['step'].total_agent_steps for p in all_profiles)

    for k, v in profile_dict.items():
        print(f'{k}: Calls={v.calls}, Elapsed={v.elapsed}, SPS={v.calls / v.elapsed}')

    step = profile_dict['step']
    print(f'Total Agent SPS: {total_agent_steps / total_time}, Average Agent SPS: {total_agent_steps / step.elapsed}, Avg Agents/Step: {total_agent_steps / step.calls}')
    pass

if __name__ == '__main__':
    steps = [1, 10, 100]
    backends = [
        #pufferlib.vectorization.serial.VecEnv,
        pufferlib.vectorization.multiprocessing.VecEnv,
        #pufferlib.vectorization.multiprocessing_shared.VecEnv,
        #pufferlib.vectorization.ray.VecEnv,
        #pufferlib.vectorization.ray_shared.VecEnv,
    ]

    binding = pufferlib.registry.nmmo.make_binding()

    for s in steps:
        for b in backends:
            sps = _test_vectorization_performance(binding, b, s, 1, 1)
            print(f'{b.__module__.split(".")[-1]}: {steps} steps: {sps} steps per second')