from pdb import set_trace as T
import numpy as np
import gym

import pufferlib.emulation
import pufferlib.vectorization

import mock_environments

MAX_AGENTS = 16
NUM_ENVS = 2


def make_gym_puffer_env():
    observation_space = gym.spaces.Box(low=-2**20, high=2**20, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    env = mock_environments.make_mock_singleagent_env(observation_space, action_space)
    env = pufferlib.emulation.GymPufferEnv(env_cls=env)
    return env

def make_pettingzoo_puffer_env():
    observation_space = gym.spaces.Box(low=-2**20, high=2**20, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    env = mock_environments.make_mock_multiagent_env(
        observation_space, action_space,
        initial_agents=MAX_AGENTS,
        max_agents=MAX_AGENTS,
        spawn_per_tick=0,
        death_per_tick=0,
    )
    env = pufferlib.emulation.PettingZooPufferEnv(env_cls=env)
    return env

def test_gym_puffer_env():
    env = make_gym_puffer_env()
    ob = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)

        if done:
            ob = env.reset()

def test_pettingzoo_puffer_env():
    env = make_pettingzoo_puffer_env()
    obs = env.reset()
    for _ in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        obs, rewards, dones, infos = env.step(actions)

def test_gym_multi_puffer_env():
    env = pufferlib.vectorization.SerialGymPufferEnvs(env_creator=make_gym_puffer_env, n=NUM_ENVS)
    obs = env.reset()
    for _ in range(10):
        actions = [env.single_action_space.sample() for _ in range(NUM_ENVS)]
        obs, rewards, dones, infos = env.step(actions) 

def test_petting_zoo_multi_puffer_env():
    env = pufferlib.vectorization.SerialPettingZooPufferEnvs(env_creator=make_pettingzoo_puffer_env, n=NUM_ENVS)
    obs = env.reset()
    for _ in range(10):
        actions  =[env.single_action_space.sample() for _ in range(NUM_ENVS*MAX_AGENTS)]
        obs, rewards, dones, infos = env.step(actions)

def test_gym_puffer_vecenv():
    env = pufferlib.vectorization.Serial(env_creator=make_gym_puffer_env, n=NUM_ENVS)
    obs = env.reset()
    for _ in range(10):
        actions = [env.single_action_space.sample() for _ in range(NUM_ENVS)]
        obs, rewards, dones, infos = env.step(actions)

def test_atari_puffer_vecenv():
    from pufferlib.registry import atari
    env = pufferlib.vectorization.Serial(env_creator=atari.make_env, env_args=['Breakout-v4', 1], n=NUM_ENVS)
    obs = env.reset()
    for _ in range(10):
        actions = [env.single_action_space.sample() for _ in range(NUM_ENVS)]
        obs, rewards, dones, infos = env.step(actions)

def test_procgen_puffer_vecenv():
    from pufferlib.registry import procgen
    envs = procgen.VecEnv(env_name='coinrun', num_envs=NUM_ENVS)
    obs = envs.reset()
    for _ in range(10):
        actions = [envs.single_action_space.sample() for _ in range(NUM_ENVS)]
        obs, rewards, dones, infos = envs.step(actions)


if __name__ == '__main__':
    test_gym_puffer_env()
    test_pettingzoo_puffer_env()
    test_gym_multi_puffer_env()
    test_gym_multi_puffer_env()
    test_gym_puffer_vecenv()
    test_procgen_puffer_vecenv()