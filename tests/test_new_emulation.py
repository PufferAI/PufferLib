from pdb import set_trace as T
import numpy as np
import gym

import pufferlib.new_emulation

import mock_environments


def test_gym_puffer_env():
    observation_space = gym.spaces.Box(low=-2**20, high=2**20, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    env = mock_environments.make_mock_singleagent_env(observation_space, action_space)
    env = pufferlib.new_emulation.GymPufferEnv(env_cls=env)

    ob = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)

        if done:
            ob = env.reset()

def test_pettingzoo_puffer_env():
    observation_space = gym.spaces.Box(low=-2**20, high=2**20, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    env = mock_environments.make_mock_multiagent_env(
        observation_space, action_space,
        initial_agents=16,
        max_agents=16,
        spawn_per_tick=0,
        death_per_tick=0,
    )
    env = pufferlib.new_emulation.PettingZooPufferEnv(env_cls=env)

    obs = env.reset()
    for _ in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    test_gym_puffer_env()
    test_pettingzoo_puffer_env()