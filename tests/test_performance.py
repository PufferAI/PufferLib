from pdb import set_trace as T

import gym
import time

from pufferlib.vecenvs import VecEnvs
from environments import bindings


class MockEnv:
    def __init__(self, delay=0):
        self.agents = [1]
        self.possible_agents = [1]
        self.done = False
        self.delay = delay

    def reset(self):
        return {1: self.observation_space(1).sample()}

    def step(self, actions):
        obs = {1: 0}
        rewards = {1: 1}
        dones = {1: False}
        infos = {1: {}}

        time.sleep(self.delay)

        return obs, rewards, dones, infos

    def observation_space(self, agent):
        return gym.spaces.Discrete(2)

    def action_space(self, agent):
        return gym.spaces.Discrete(2)

class MockBinding:
    def __init__(self, delay):
        self.env_creator = lambda: MockEnv(delay)

    @property
    def single_observation_space(self):
        return gym.spaces.Discrete(2)

    @property
    def single_action_space(self):
        return gym.spaces.Discrete(2)

def test_mock_speed(steps=100, num_workers=8, envs_per_worker=32, delay=0):
    env = MockEnv(delay)
    env.reset()

    total_time = 0
    for i in range(steps):
        start = time.time()
        env.step(None)
        total_time += time.time() - start
    raw_sps = steps / total_time

    binding = MockBinding(delay)

    envs = VecEnvs(
        binding,
        num_workers=num_workers,
        envs_per_worker=envs_per_worker,
    )
    envs.reset()

    actions = [1] * num_workers * envs_per_worker

    start = time.time()
    for i in range(steps):
        envs.step(actions)

    total_time = time.time() - start
    vec_sps = steps / total_time

    print(f'MockEnv SPS Raw/Vec/Speedup): {int(raw_sps)}/{int(vec_sps)}/{vec_sps/raw_sps}')

def test_env_speedup(steps=100, num_workers=8):
    #for binding in bindings:
    for binding in [bindings[2]]:
        # Test Env SPS
        env = binding.env_creator()
        obs = env.reset() 

        #Set envs per worker by number of agents
        envs_per_worker = max(1, 32 // len(env.possible_agents))
        total_steps = steps * num_workers * envs_per_worker

        actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}

        total_time = 0
        for i in range(steps):
            atns = {agent: env.action_space(agent).sample() for agent in obs}
            start = time.time()
            obs, _, dones, _ = env.step(atns)
            if all(dones.values()):
                obs = env.reset()
            total_time += time.time() - start

        raw_sps = total_steps / total_time

        # Test VecEnv SPS
        envs = VecEnvs(
            binding,
            num_workers=num_workers,
            envs_per_worker=envs_per_worker,
        )
        obs = envs.reset()

        total_time = 0
        for i in range(steps):
            actions = [binding.single_action_space.sample() for _ in range(len(env.possible_agents) * num_workers * envs_per_worker)]
            start = time.time()
            obs, _, dones, _ = envs.step(actions)
            total_time += time.time() - start
        vec_sps = total_steps / total_time

        print(f'{binding.env_name} SPS Raw/Vec/Speedup): {int(raw_sps)}/{int(vec_sps)}/{vec_sps/raw_sps}')

if __name__ == '__main__':
    test_mock_speed(steps=100, delay=0)
    test_mock_speed(steps=100, delay=0.01)
    test_mock_speed(steps=10, delay=0.1)
    test_env_speedup()