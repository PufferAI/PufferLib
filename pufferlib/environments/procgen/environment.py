from pdb import set_trace as T
import numpy as np

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.environments


def env_creator():
    procgen = pufferlib.environments.try_import('procgen') 
    return procgen.ProcgenEnv

    #return gym.make
    import gym3
    from procgen.env import ProcgenGym3Env
    return ProcgenGym3Env

def make_env(name='bigfish', num_envs=24, num_levels=0,
        start_level=0, distribution_mode='easy'):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
    assert int(num_envs) == float(num_envs), "num_envs must be an integer"
    num_envs = int(num_envs)
    envs = env_creator()(
        env_name=name,
        num_envs=num_envs,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = ProcgenPettingZooEnv(envs, num_envs)
    return pufferlib.emulation.PettingZooPufferEnv(
        env=envs,
        postprocessor_cls=ProcgenPostprocessor,
    )

class ProcgenPettingZooEnv:
    '''Fakes a multiagent interface to ProcGen where each env
    is an agent. Very low overhead.'''
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs
        self.possible_agents = list(range(num_envs))
        self.agents = self.possible_agents

    def observation_space(self, agent):
        return self.env.observation_space

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

class ProcgenPostprocessor(pufferlib.emulation.Postprocessor):
    def reset(self, obs):
        self.epoch_return = 0
        self.epoch_length = 0

    def reward_done_truncated_info(self, reward, done, truncated, info):
        if isinstance(reward, (list, np.ndarray)):
            reward = sum(reward.values())

        self.epoch_length += 1
        self.epoch_return += reward

        if done or truncated:
            info['return'] = self.epoch_return
            info['length'] = self.epoch_length
            self.epoch_return = 0
            self.epoch_length = 0

        return reward, done, truncated, info
