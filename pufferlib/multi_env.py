from pdb import set_trace as T
from collections.abc import Mapping
import numpy as np

import pufferlib.exceptions


def create_precheck(env_creator, env_args, env_kwargs):
    if env_args is None:
        env_args = []
    if env_kwargs is None:
        env_kwargs = {}

    if not callable(env_creator):
        raise pufferlib.exceptions.APIUsageError('env_creator must be callable')
    if not isinstance(env_args, list):
        raise pufferlib.exceptions.APIUsageError('env_args must be a list')
    # TODO: port namespace to Mapping
    if not isinstance(env_kwargs, Mapping):
        raise pufferlib.exceptions.APIUsageError('env_kwargs must be a dictionary or None')

    return env_args, env_kwargs

def __init__(self, env_creator: callable = None, env_args: list = [],
        env_kwargs: dict = {}, n: int = 1):
    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    self.envs = [env_creator(*env_args, **env_kwargs) for _ in range(n)]
    self.preallocated_obs = None

def put(state, *args, **kwargs):
    for e in state.envs:
        e.put(*args, **kwargs)
    
def get(state, *args, **kwargs):
    return [e.get(*args, **kwargs) for e in state.envs]

def close(state):
    for env in state.envs:
        env.close()

class GymnasiumMultiEnv:
    __init__ = __init__
    put = put
    get = get
    close = close

    def reset(self, seed=None):
        if self.preallocated_obs is None:
            obs_space = self.envs[0].observation_space
            obs_n = obs_space.shape[0]
            n_envs = len(self.envs)

            self.preallocated_obs = np.empty(
                (n_envs, *obs_space.shape), dtype=obs_space.dtype)
            self.preallocated_rewards = np.empty(n_envs, dtype=np.float32)
            self.preallocated_dones = np.empty(n_envs, dtype=bool)
            self.preallocated_truncateds = np.empty(n_envs, dtype=bool)

        infos = []
        for idx, e in enumerate(self.envs):
            if seed is None:
                ob, i = e.reset()
            else:
                ob, i = e.reset(seed=hash(1000*seed + idx))

            i['mask'] = True
            infos.append(i)
            self.preallocated_obs[idx] = ob

        self.preallocated_rewards[:] = 0
        self.preallocated_dones[:] = False
        self.preallocated_truncateds[:] = False

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos)

    def step(self, actions):
        infos = []
        for idx, (env, atn) in enumerate(zip(self.envs, actions)):
            if env.done:
                o, i = env.reset()
                self.preallocated_rewards[idx] = 0
                self.preallocated_dones[idx] = False
                self.preallocated_truncateds[idx] = False
            else:
                o, r, d, t, i = env.step(atn)
                self.preallocated_rewards[idx] = r
                self.preallocated_dones[idx] = d
                self.preallocated_truncateds[idx] = t

            i['mask'] = True
            infos.append(i)
            self.preallocated_obs[idx] = o

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos)

class PettingZooMultiEnv:
    __init__ = __init__
    put = put
    get = get
    close = close

    def reset(self, seed=None):
        if self.preallocated_obs is None:
            obs_space = self.envs[0].single_observation_space
            obs_n = obs_space.shape[0]
            n_agents = len(self.envs[0].possible_agents)
            n_envs = len(self.envs)
            n = n_envs * n_agents

            self.preallocated_obs = np.empty(
                (n, *obs_space.shape), dtype=obs_space.dtype)
            self.preallocated_rewards = np.empty(n, dtype=np.float32)
            self.preallocated_dones = np.empty(n, dtype=bool)
            self.preallocated_truncateds = np.empty(n, dtype=bool)

        self.agent_keys = []
        infos = []
        ptr = 0
        for idx, e in enumerate(self.envs):
            if seed is None:
                obs, i = e.reset()
            else:
                obs, i = e.reset(seed=hash(1000*seed + idx))

            self.agent_keys.append(list(obs.keys()))
            infos.append(i)

            for o in obs.values():
                self.preallocated_obs[ptr] = o
                ptr += 1

        self.preallocated_rewards[:] = 0
        self.preallocated_dones[:] = False
        self.preallocated_truncateds[:] = False

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos)

    def step(self, actions):
        actions = np.array_split(actions, len(self.envs))
        rewards, dones, truncateds, infos = [], [], [], []

        ptr = 0
        n_envs = len(self.envs)
        n_agents = len(self.envs[0].possible_agents)
        assert n_envs == len(self.agent_keys) == len(actions)

        for idx in range(n_envs):
            a_keys, env, atns = self.agent_keys[idx], self.envs[idx], actions[idx]
            start = idx * n_agents
            end = start + n_agents
            if env.done:
                o, i = env.reset()
                self.preallocated_rewards[start:end] = 0
                self.preallocated_dones[start:end] = False
                self.preallocated_truncateds[start:end] = False
            else:
                assert len(a_keys) == len(atns)
                atns = dict(zip(a_keys, atns))
                o, r, d, t, i = env.step(atns)
                self.preallocated_rewards[start:end] = list(r.values())
                self.preallocated_dones[start:end] = list(d.values())
                self.preallocated_truncateds[start:end] = list(t.values())

            infos.append(i)
            self.agent_keys[idx] = list(o.keys())

            for oo in o.values():
                self.preallocated_obs[ptr] = oo
                ptr += 1

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos)
