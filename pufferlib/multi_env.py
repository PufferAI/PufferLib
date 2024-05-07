from pdb import set_trace as T
from collections.abc import Mapping
import numpy as np

import pufferlib.exceptions
import pufferlib.emulation


class PufferEnv:
    #@property
    #def num_agents(self):
    #    raise NotImplementedError

    #@property
    #def observation_space(self):
    #    raise NotImplementedError

    #@property
    #def action_space(self):
    #    raise NotImplementedError

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def close(self):
        pass

class PufferEnvWrapper(PufferEnv):
    def __init__(self, env_creator: callable = None, env_args: list = [],
            env_kwargs: dict = {}, n: int = 1):
        assert n > 0
        env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
        envs = [env_creator(*env_args, **env_kwargs) for _ in range(n)]
        self.envs = envs

        # Check that all envs are either Gymnasium or PettingZoo
        is_gymnasium = all(isinstance(e, pufferlib.emulation.GymnasiumPufferEnv) for e in envs)
        is_pettingzoo = all(isinstance(e, pufferlib.emulation.PettingZooPufferEnv) for e in envs)
        #is_puffer = all(isinstance(e, PufferEnv) for e in envs)
        #assert is_gymnasium or is_pettingzoo or is_puffer
        self.is_gymnasium = is_gymnasium
        self.is_pettingzoo = is_pettingzoo

        # Check that all envs have the same observation and action spaces
        # TODO: additional check here

        env = envs[0]
        if is_gymnasium:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.agents_per_env = 1
            self.num_agents = len(envs)
        elif is_pettingzoo:
            self.observation_space = envs.single_observation_space
            self.action_space = envs.single_action_space
            self.agents_per_env = len(envs.possible_agents)
            self.num_agents = len(envs) * self.agents_per_env
        else:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.agents_per_env = env.num_agents
            self.num_agents = len(envs) * self.agents_per_env

        self.preallocated_obs = np.empty(
            (self.num_agents, *self.observation_space.shape), dtype=self.observation_space.dtype)
        self.preallocated_rewards = np.empty(self.num_agents, dtype=np.float32)
        self.preallocated_dones = np.empty(self.num_agents, dtype=bool)
        self.preallocated_truncateds = np.empty(self.num_agents, dtype=bool)
        self.preallocated_masks = np.ones(self.num_agents, dtype=bool)

    def reset(self, seed=None):
        infos = []
        ptr = 0
        for idx, env in enumerate(self.envs):
            if seed is None:
                ob, i = env.reset()
            else:
                ob, i = env.reset(seed=hash(1000*seed + idx))

            if self.is_pettingzoo:
                ob = list(ob.values())
                i = list(i.values())
                
            infos.append(i)
            end = ptr + self.agents_per_env
            self.preallocated_obs[ptr:end] = ob
            ptr = end

        self.preallocated_rewards[:] = 0
        self.preallocated_dones[:] = False
        self.preallocated_truncateds[:] = False
        self.preallocated_masks[:] = 1

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos)

    def step(self, actions):
        rewards, dones, truncateds, infos = [], [], [], []

        ptr = 0
        for idx, env in enumerate(self.envs):
            end = ptr + self.agents_per_env
            atns = actions[ptr:end]

            if self.is_gymnasium:
                atns = atns[0]
            elif self.is_pettingzoo:
                atns = dict(zip(env.possible_agents, atns))

            if env.done:
                o, i = env.reset()
                r = 0
                d = False
                t = False
            else:
                o, r, d, t, i = env.step(atns)

            if self.is_pettingzoo:
                o = list(o.values())
                r = list(r.values())
                d = list(d.values())
                t = list(t.values())
                i = list(i.values())

                self.preallocated_masks[ptr:end] = list(env.mask.values())

                # Delete empty keys
                i = [e for e in i if e]
                infos.extend(i)
            else:
                infos.append(i)

            self.preallocated_obs[ptr:end] = o
            self.preallocated_rewards[ptr:end] = r
            self.preallocated_dones[ptr:end] = d
            self.preallocated_truncateds[ptr:end] = t
            self.ptr = end

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos)

class PufferEnvWrapper(PufferEnv):
    def __init__(self, env_creator: callable = None, env_args: list = [],
            env_kwargs: dict = {}, n: int = 1, obs_mem=None, rew_mem=None, done_mem=None, trunc_mem=None, mask_mem=None):
        assert n > 0
        env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
        envs = [env_creator(*env_args, **env_kwargs) for _ in range(n)]
        self.envs = envs

        # Check that all envs are either Gymnasium or PettingZoo
        is_gymnasium = all(isinstance(e, pufferlib.emulation.GymnasiumPufferEnv) for e in envs)
        is_pettingzoo = all(isinstance(e, pufferlib.emulation.PettingZooPufferEnv) for e in envs)
        #is_puffer = all(isinstance(e, PufferEnv) for e in envs)
        #assert is_gymnasium or is_pettingzoo or is_puffer
        self.is_gymnasium = is_gymnasium
        self.is_pettingzoo = is_pettingzoo

        # Check that all envs have the same observation and action spaces
        # TODO: additional check here

        env = envs[0]
        if is_gymnasium:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.agents_per_env = 1
            self.num_agents = len(envs)
        elif is_pettingzoo:
            self.observation_space = envs.single_observation_space
            self.action_space = envs.single_action_space
            self.agents_per_env = len(envs.possible_agents)
            self.num_agents = len(envs) * self.agents_per_env
        else:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.agents_per_env = env.num_agents
            self.num_agents = self.agents_per_env


        if obs_mem is None:
            self.preallocated_obs = np.empty(
                (n, self.num_agents, *self.observation_space.shape), dtype=self.observation_space.dtype)
        else:
            self.preallocated_obs = obs_mem.reshape(n, self.num_agents, *self.observation_space.shape)
        if rew_mem is None:
            self.preallocated_rewards = np.empty((n, self.num_agents), dtype=np.float32)
        else:
            self.preallocated_rewards = rew_mem.reshape(n, self.num_agents)
        if done_mem is None:
            self.preallocated_dones = np.empty((n, self.num_agents), dtype=bool)
        else:
            self.preallocated_dones = done_mem.reshape(n, self.num_agents)
        if trunc_mem is None:
            self.preallocated_truncateds = np.empty((n, self.num_agents), dtype=bool)
        else:
            self.preallocated_truncateds = trunc_mem.reshape(n, self.num_agents)
        if mask_mem is None:
            self.preallocated_masks = np.ones((n, self.num_agents), dtype=bool)
        else:
            self.preallocated_masks = mask_mem.reshape(n, self.num_agents)

        #self.preallocated_rewards = np.empty(self.num_agents, dtype=np.float32)
        #self.preallocated_dones = np.empty(self.num_agents, dtype=bool)
        #self.preallocated_truncateds = np.empty(self.num_agents, dtype=bool)
        #self.preallocated_masks = np.ones(self.num_agents, dtype=bool)

    def reset(self, seed=None):
        infos = []
        for idx, env in enumerate(self.envs):
            if seed is None:
                ob, i = env.reset()
            else:
                ob, i = env.reset(seed=hash(1000*seed + idx))

            if self.is_pettingzoo:
                ob = list(ob.values())
                i = list(i.values())
                
            infos.append(i)
            self.preallocated_obs[idx] = ob

        self.preallocated_rewards[:] = 0
        self.preallocated_dones[:] = False
        self.preallocated_truncateds[:] = False
        self.preallocated_masks[:] = 1

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos, self.preallocated_masks)

    def step(self, actions):
        rewards, dones, truncateds, infos = [], [], [], []

        for idx, env in enumerate(self.envs):
            atns = actions[idx]

            if self.is_gymnasium:
                atns = atns[0]
            elif self.is_pettingzoo:
                atns = dict(zip(env.possible_agents, atns))

            if env.done:
                o, i = env.reset()
                r = 0
                d = False
                t = False
            else:
                o, r, d, t, i = env.step(atns)

            if self.is_pettingzoo:
                o = list(o.values())
                r = list(r.values())
                d = list(d.values())
                t = list(t.values())
                i = list(i.values())

                self.preallocated_masks[idx] = list(env.mask.values())

                # Delete empty keys
                i = [e for e in i if e]
                infos.extend(i)
            else:
                infos.append(i)

            self.preallocated_obs[idx] = o
            self.preallocated_rewards[idx] = r
            self.preallocated_dones[idx] = d
            self.preallocated_truncateds[idx] = t

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos, self.preallocated_masks)


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
        env_kwargs: dict = {}, n: int = 1, mask_agents=False):
    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    self.envs = [env_creator(*env_args, **env_kwargs) for _ in range(n)]
    self.preallocated_obs = None
    self.mask_agents = mask_agents

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
            self.preallocated_masks = np.ones(n_envs, dtype=bool)

        infos = []
        for idx, e in enumerate(self.envs):
            if seed is None:
                ob, i = e.reset()
            else:
                ob, i = e.reset(seed=hash(1000*seed + idx))

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
            self.preallocated_masks = np.empty(n, dtype=bool)

        self.agent_keys = []
        infos = []
        ptr = 0
        for idx, e in enumerate(self.envs):
            if seed is None:
                obs, i = e.reset()
            else:
                obs, i = e.reset(seed=hash(1000*seed + idx))

            self.agent_keys.append(list(obs.keys()))

            # Delete empty keys
            for k in list(i):
                if not i[k]:
                    del i[k]

            infos.append(i)

            for o, m in zip(obs.values(), e.mask.values()):
                self.preallocated_obs[ptr] = o
                self.preallocated_masks[ptr] = m
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
                self.preallocated_masks[start:end] = list(env.mask.values())

            # Delete empty keys
            for k in list(i):
                if not i[k]:
                    del i[k]

            infos.append(i)
            self.agent_keys[idx] = list(o.keys())

            for oo in o.values():
                self.preallocated_obs[ptr] = oo
                ptr += 1

        return (self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, infos)
