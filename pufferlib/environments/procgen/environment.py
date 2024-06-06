from pdb import set_trace as T
import numpy as np

import gym
import shimmy
import functools

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.postprocess


def env_creator(name='bigfish'):
    return functools.partial(make, name)

def make(name, num_envs=1, num_levels=0,
        start_level=0, distribution_mode='easy'):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    assert int(num_envs) == float(num_envs), "num_envs must be an integer"
    num_envs = int(num_envs)

    procgen = pufferlib.environments.try_import('procgen') 
    envs = procgen.ProcgenEnv(
        env_name=name,
        num_envs=num_envs,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        render_mode='rgb_array',
    )
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = ProcgenGymnasiumEnv(envs)
    #envs = ProcgenPettingZooEnv(envs, num_envs)
    #envs = pufferlib.postprocess.MultiagentEpisodeStats(envs)
    envs = pufferlib.postprocess.EpisodeStats(envs)
    return pufferlib.emulation.GymnasiumPufferEnv(env=envs)
    #return pufferlib.emulation.PettingZooPufferEnv(env=envs)

class ProcgenGymnasiumEnv:
    '''Fakes a multiagent interface to ProcGen where each env
    is an agent. Very low overhead.'''
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space['rgb']
        self.action_space = self.env.action_space

    @property
    def render_mode(self):
        return 'rgb_array'

    def reset(self, seed=None):
        obs = self.env.reset()[0]
        return obs, {}

    def render(self):
        return self.env.env.env.env.env.env.observe()[1]['rgb'][0]
        return self.env.env.env.env.env.env.get_info()[0]['rgb']

    def close(self):
        return self.env.close()

    def step(self, actions):
        actions = np.asarray(actions).reshape(1)
        obs, rewards, dones, infos = self.env.step(actions)
        return obs[0], rewards[0], dones[0], False, infos[0]

class ProcgenPettingZooEnv:
    '''Fakes a multiagent interface to ProcGen where each env
    is an agent. Very low overhead.'''
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.possible_agents = list(range(num_envs))
        self.agents = self.possible_agents

    def observation_space(self, agent):
        return self.env.observation_space['rgb']

    def action_space(self, agent):
        return self.env.action_space

    def reset(self, seed=None):
        obs = self.env.reset()
        obs = {i: o for i, o in enumerate(obs)}
        info = {i: {'mask': True} for i in obs}
        return obs, info

    def step(self, actions):
        actions = np.array([actions[i] for i in range(self.num_envs)])
        obs, rewards, dones, infos = self.env.step(actions)
        obs = {i: o for i, o in enumerate(obs)}
        rewards = {i: r for i, r in enumerate(rewards)}
        dones = {i: bool(d) for i, d in enumerate(dones)}
        truncateds = {i: False for i in range(len(obs))}
        infos = {i: {'mask': True} for i in range(len(obs))}
        return obs, rewards, dones, truncateds, infos
