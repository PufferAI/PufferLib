from pdb import set_trace as T
import numpy as np

import random
import time

import gym
from stable_baselines3.common.vec_env import SubprocVecEnv

from pettingzoo.utils.env import ParallelEnv

import pufferlib
import pufferlib.utils
import pufferlib.binding
import pufferlib.registry
from pufferlib.vecenvs import VecEnvs


class MockEnv(gym.Env):
    def __init__(self, delay_iterations, ob_dim):
        self.delay_iterations = delay_iterations
        self.ob_dim = ob_dim


    def seed(self, seed):
        self.ob = seed + np.zeros(self.ob_dim)

    def reset(self):
        return self.ob

    @property 
    def observation_space(self):
        return gym.spaces.Box(
            low=-2**20, high=2**20,
            shape=(self.ob_dim,), dtype=np.float32,
        )

    @property 
    def action_space(self):
        return gym.spaces.Discrete(1)

    def step(self, action):
        inc = 0
        for i in range(self.delay_iterations):
            inc += 1
    
        return self.ob, 0, 0, {}

def test_envs(env_name, env_creator, steps=1000):
    binding = pufferlib.binding.auto(
        env=env_creator,
        env_name=env_name,
    )

    env = env_creator()
    env.reset()

    env_timer = pufferlib.utils.Profiler()
    for i in range(steps):
        with env_timer:
            env.step(0)

    print('Env SPS: ', steps / env_timer.elapsed)

def test_vec_envs(env_name, single_env_creator, multi_env_creator):
    multidiscrete_atns = [[0] for _ in range(4)]

    # PufferLib VecEnvs
    import ray
    ray.shutdown()
    ray.init(include_dashboard=False, ignore_reinit_error=True)

    #single_binding = pufferlib.binding.auto(
    #    env=single_env_creator,
    #    env_name=env_name,
    #)
    single_binding = pufferlib.registry.Atari('BreakoutNoFrameskip-v4')

    single_puffer_envs = VecEnvs(single_binding, num_workers=4, envs_per_worker=1)
    single_ob = single_puffer_envs.reset()
    print(single_ob)
    print(single_ob.sum(1))

    #multi_binding = pufferlib.binding.auto(
    #    env=multi_env_creator,
    #    env_name=env_name,
    #)
    multi_binding = pufferlib.registry.Atari('BreakoutNoFrameskip-v4')

    multi_puffer_envs = VecEnvs(multi_binding, num_workers=2, envs_per_worker=2)
    multi_ob = multi_puffer_envs.reset()

    print(multi_ob)
    print(multi_ob.sum(1))


if __name__ == '__main__':
    class Counter:
        def __init__(self):
            self.i = 0

        def __call__(self):
            self.i += 1
            return self.i

    single_counter = Counter()
    multi_counter = Counter()

    single_env_creator = lambda: MockEnv(delay_iterations=0, ob_dim=1)
    multi_env_creator = lambda: MockEnv(delay_iterations=0, ob_dim=1)

    test_vec_envs('MockEnv', single_env_creator, multi_env_creator)