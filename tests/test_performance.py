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


class MockEnv:
    def __init__(self, delay=0, bandwith=1):
        self.agents = [1]
        self.possible_agents = [1]
        self.done = False

        self.delay = delay
        self.bandwidth = bandwith

    def reset(self, seed=None):
        return {1: self.observation_space(1).sample()}

    def step(self, actions):
        obs = {1: 0}
        rewards = {1: 1}
        dones = {1: False}
        infos = {1: {}}

        time.sleep(self.delay)

        return obs, rewards, dones, infos

    def observation_space(self, agent):
        return gym.spaces.Box(
            low=-2**20, high=2**20,
            shape=(self.bandwidth,), dtype=np.float32
        )

    def action_space(self, agent):
        return gym.spaces.Discrete(2)

class MockBinding:
    def __init__(self, delay, bandwith):
        self.env_creator = lambda: MockEnv(delay, bandwith)
        self.bandwidth = bandwith

    @property
    def single_observation_space(self):
        return gym.spaces.Box(
            low=-2**20, high=2**20,
            shape=(self.bandwidth,), dtype=np.float32
        )

    @property
    def single_action_space(self):
        return gym.spaces.Discrete(2)

def _test_performance(binding, vec_backend, steps, num_workers, envs_per_worker):
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
            binding = MockBinding(delay=d, bandwith=100_000)
            sps = _test_performance(binding, b, steps, 2, 1)
            print(f'{b.__module__.split(".")[-1]}: {steps} steps with {d} delay: {sps} steps per second')

    #import pufferlib.registry.atari
    #binding = pufferlib.registry.atari.make_binding('BreakoutNoFrameskip-v4', framestack=1)