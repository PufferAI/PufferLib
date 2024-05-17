from pdb import set_trace as T
from collections.abc import Mapping
import numpy as np

import pufferlib.environment
import pufferlib.exceptions
import pufferlib.emulation


class PufferEnvWrapper(pufferlib.environment.PufferEnv):
    def __init__(self, env_creator: callable = None, env_args: list = [],
            env_kwargs: dict = {}, n: int = 1, obs_mem=None, rew_mem=None, done_mem=None, trunc_mem=None, mask_mem=None):
        if n < 1:
            raise pufferlib.exceptions.APIUsageError('n (environments) must be at least 1')

        envs = [env_creator(*env_args, **env_kwargs) for _ in range(n)]
        self.envs = envs

        # Check that all envs are either Gymnasium or PettingZoo
        is_gymnasium = all(isinstance(e, pufferlib.emulation.GymnasiumPufferEnv) for e in envs)
        is_pettingzoo = all(isinstance(e, pufferlib.emulation.PettingZooPufferEnv) for e in envs)
        is_puffer = all(isinstance(e, pufferlib.environment.PufferEnv) for e in envs)
        assert is_gymnasium or is_pettingzoo or is_puffer
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
                (n, self.agents_per_env, *self.observation_space.shape), dtype=self.observation_space.dtype)
        else:
            self.preallocated_obs = obs_mem.reshape(n, self.agents_per_env, *self.observation_space.shape)
        if rew_mem is None:
            self.preallocated_rewards = np.empty((n, self.agents_per_env), dtype=np.float32)
        else:
            self.preallocated_rewards = rew_mem.reshape(n, self.agents_per_env)
        if done_mem is None:
            self.preallocated_dones = np.empty((n, self.agents_per_env), dtype=bool)
        else:
            self.preallocated_dones = done_mem.reshape(n, self.agents_per_env)
        if trunc_mem is None:
            self.preallocated_truncateds = np.empty((n, self.agents_per_env), dtype=bool)
        else:
            self.preallocated_truncateds = trunc_mem.reshape(n, self.agents_per_env)
        if mask_mem is None:
            self.preallocated_masks = np.ones((n, self.agents_per_env), dtype=bool)
        else:
            self.preallocated_masks = mask_mem.reshape(n, self.agents_per_env)

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

    def put(self, *args, **kwargs):
        for e in self.envs:
            e.put(*args, **kwargs)
        
    def get(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.envs]

    def close(self):
        for env in self.envs:
            env.close()
