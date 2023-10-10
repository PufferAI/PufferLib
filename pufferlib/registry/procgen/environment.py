from pdb import set_trace as T
import numpy as np

import gym
import shimmy

import pufferlib
import pufferlib.emulation
import pufferlib.registry


def env_creator():
    procgen = pufferlib.registry.try_import('procgen') 
    return procgen.ProcgenEnv

    #return gym.make
    import gym3
    from procgen.env import ProcgenGym3Env
    return ProcgenGym3Env

def make_env(name='bigfish', num_envs=1, num_levels=0,
        start_level=0, distribution_mode='easy'):
    '''Atari creation function with default CleanRL preprocessing based on Stable Baselines3 wrappers'''
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
    return ProcgenPettingZooEnv(env=envs)

    env = env_creator()(
        num=1,
        env_name=name,
        distribution_mode=distribution_mode,
    )

    # Note: CleanRL normalizes and clips rewards
    import gym3
    env = gym3.ToGymEnv(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeReward(env, gamma=0.999)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env = shimmy.GymV21CompatibilityV0(env=env)
    env = pufferlib.emulation.GymPufferEnv(
        env=env,
        postprocessor_cls=ProcgenPostprocessor,
    )
    return env

class ProcgenPettingZooEnv(pufferlib.emulation.PettingZooPufferEnv):
    '''Fakes a multiagent interface to ProcGen where each env
    is an agent. Very low overhead.'''
    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None):
        obs = self.env.reset()
        obs = {i: o for i, o in enumerate(obs)}
        return obs, {}

    def step(self, actions):
        actions = [actions[i] for i in range(len(actions))]
        obs, rewards, dones, infos = self.env.step(actions)
        obs = {i: o for i, o in enumerate(obs)}
        rewards = {i: r for i, r in enumerate(rewards)}
        dones = {i: d for i, d in enumerate(dones)}
        truncateds = {i: False for i in range(len(obs))}
        infos = {i: {} for i in range(len(obs))}
        return obs, rewards, dones, truncateds, infos

class ProcgenPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs):
        try:
            return obs['rgb']
        except:
            return obs

    def reward_done_truncated_info(self, reward, done, truncated, info):
        return float(reward), bool(done), truncated, info

class ProcgenVecEnv:
    '''WIP Vectorized Procgen environment wrapper
    
    Does not use normal PufferLib emulation'''
    def __init__(self, env_name, num_envs=1,
            num_levels=0, start_level=0,
            distribution_mode="easy"):

        self.num_envs = num_envs
        self.envs = ProcgenEnv(
            env_name=env_name,
            num_envs=num_envs,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
        )

    @property
    def single_observation_space(self):
        return self.envs.observation_space['rgb']

    @property
    def single_action_space(self):
        return self.envs.action_space

    def reset(self, seed=None):
        obs = self.envs.reset()['rgb']
        rewards = [0] * self.num_envs
        dones = [False] * self.num_envs
        infos = [{}] * self.num_envs
        return obs, rewards, dones, infos

    def step(self, actions):
        actions = np.array(actions)
        obs, rewards, dones, infos = self.envs.step(actions)
        return obs['rgb'], rewards, dones, infos
