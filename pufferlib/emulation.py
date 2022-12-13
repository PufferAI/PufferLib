from pdb import set_trace as T

import numpy as np

from collections import OrderedDict
from collections.abc import MutableMapping
import functools
import inspect

import gym
import pettingzoo
from pettingzoo.utils.env import ParallelEnv

from pufferlib import utils


def PufferEnv(Env, 
        feature_parser=None,
        reward_shaper=None,
        rllib_dones=False,
        emulate_flat_obs=True,
        emulate_flat_atn=True,
        emulate_const_horizon=1024,
        emulate_const_num_agents=128,
        obs_dtype=np.float32):

    # Consider integrating these?
    #env = wrappers.AssertOutOfBoundsWrapper(env)
    #env = wrappers.OrderEnforcingWrapper(env)
    class PufferWrapper(ParallelEnv):
        def __init__(self, *args, **kwargs):
            # Infer obs space from first agent
            # Assumes all agents have the same obs space
            self.dummy_obs = {}
            self._step = 0
            self.done = False
            self.obs_dtype = obs_dtype

            self.feature_parser = feature_parser
            self.reward_shaper = reward_shaper
            self.rllib_dones = rllib_dones

            self.emulate_flat_obs = emulate_flat_obs
            self.emulate_flat_atn = emulate_flat_atn
            self.emulate_const_horizon = emulate_const_horizon
            self.emulate_const_num_agents = emulate_const_num_agents
            self.emulate_multiagent = not utils.is_multiagent(Env)

            if inspect.isclass(Env):
                self.env = Env(*args, **kwargs)
            else:
                self.env = Env

            # Standardize property vs method obs/atn space interface
            if self.emulate_multiagent:
                self.agents = [1]
                self.possible_agents = [1]
            else:
                self.possible_agents = self.env.possible_agents

        def action_space(self, agent):
            '''Neural MMO Action Space

            Args:
                agent: Agent ID

            Returns:
                actions: gym.spaces object contained the structured actions
                for the specified agent. Each action is parameterized by a list
                of discrete-valued arguments. These consist of both fixed, k-way
                choices (such as movement direction) and selections from the
                observation space (such as targeting)'''
            # Get single/multiagent action space
            if self.emulate_multiagent:
                atn_space = self.env.action_space
            else:
                atn_space = self.env.action_space(agent)

            if self.emulate_flat_atn:
                assert type(atn_space) in (gym.spaces.Dict, gym.spaces.Discrete, gym.spaces.MultiDiscrete)
                if type(atn_space) == gym.spaces.Dict:
                    return pack_atn_space(atn_space)
                elif type(atn_space) == gym.spaces.Discrete:
                    return gym.spaces.MultiDiscrete([atn_space.n])

            return atn_space

        def structured_observation_space(self, agent: int):
            if self.feature_parser:
                return self.feature_parser.spec
            return self.observation_space(agent)

        def observation_space(self, agent: int):
            # Get single/multiagent observation space
            if self.emulate_multiagent:
                obs_space = self.env.observation_space
            else:
                obs_space = self.env.observation_space(agent)

            if agent not in self.dummy_obs:
                self.dummy_obs[agent] = _zero(obs_space.sample())

            dummy = self.dummy_obs[agent]

            if self.feature_parser:
                dummy = self.feature_parser({agent: dummy}, self._step)[agent]

            if self.emulate_flat_obs:
                dummy = flatten_ob(dummy, self.obs_dtype)

            shape = dummy.shape

            return gym.spaces.Box(
                low=-2**20, high=2**20,
                shape=shape, dtype=self.obs_dtype
            )

        def _process_obs(self, obs):
            # Faster to have feature parser on env but then 
            # you have to somehow mod the space to pack unpack the modded feat
            # space instead of just using the orig and featurizing in the net
            if self.emulate_const_num_agents:
                for k in self.dummy_obs:
                    if k not in obs:                                                  
                        obs[k] = self.dummy_obs[k]
 
            if self.feature_parser:
                obs = self.feature_parser(obs, self._step)

            if self.emulate_flat_obs:
                obs = pack_obs(obs, self.obs_dtype)

            return obs

        def reset(self):
            self.reset_calls_step = False
            obs = self.env.reset()

            if self.emulate_multiagent:
                obs = {1: obs}
            else:
                self.agents = self.env.agents

            self.done = False

            # Some envs implement reset by calling step
            if not self.reset_calls_step:
                obs = self._process_obs(obs)

            self._step = 0
            return obs

        def step(self, actions, **kwargs):
            assert not self.done, 'step after done'
            self.reset_calls_step = True

            # Action shape test
            if __debug__:
                for agent, atns in actions.items():
                    assert self.action_space(agent).contains(atns)

            # Unpack actions
            if self.emulate_flat_atn:
                for k, v in actions.items():
                    if self.emulate_multiagent:
                        orig_atn_space = self.env.action_space
                    else:
                        orig_atn_space = self.env.action_space(k)

                    if type(orig_atn_space) == gym.spaces.Discrete:
                        actions[k] = v[0]
                    else:
                        actions[k] = _unflatten(v, orig_atn_space)

            if self.emulate_multiagent:
                action = actions[1]
                ob, reward, done, info = self.env.step(action)
                obs = {1: ob}
                rewards = {1: reward}
                dones = {1: done}
                infos = {1: info}
            else:
                obs, rewards, dones, infos = self.env.step(actions)
                self.agents = self.env.agents

            assert '__all__' not in dones, 'Base env should not return __all__'
            self._step += 1
          
            obs = self._process_obs(obs)

            if self.reward_shaper:
                rewards = self.reward_shaper(rewards, self._step)

            # Computed before padding dones. False if no agents
            all_done = len(dones) and all(dones.values())
            self.done = all_done

            # Pad rewards/dones/infos
            if self.emulate_const_num_agents:
                for k in self.dummy_obs:
                    if k not in rewards:                                                  
                        rewards[k] = 0                                                 
                        infos[k] = {}
                        dones[k] = False

            # Terminate episode at horizon or if all agents are done
            if self.emulate_const_horizon:
                assert self._step <= self.emulate_const_horizon
                if self._step == self.emulate_const_horizon:
                    self.done = True

                for agent in dones:
                    dones[agent] = (self._step == self.emulate_const_horizon) or all_done

            # RLlib done compatibility
            if self.rllib_dones and self.done:
                dones['__all__'] = True

            # Observation shape test
            if __debug__:
                for agent, ob in obs.items():
                    assert self.observation_space(agent).contains(ob)

            return obs, rewards, dones, infos

    return PufferWrapper

def _zero(ob):
    if type(ob) == np.ndarray:
        ob *= 0
    elif type(ob) in (dict, OrderedDict):
        for k, v in ob.items():
            _zero(ob[k])
    else:
        for v in ob:
            _zero(v)
    return ob

def _flatten(nested_dict, parent_key=None):
    if parent_key is None:
        parent_key = []

    items = []
    if isinstance(nested_dict, MutableMapping) or isinstance(nested_dict, gym.spaces.Dict):
        for k, v in nested_dict.items():
            new_key = parent_key + [k]
            items.extend(_flatten(v, new_key).items())
    elif parent_key:
        items.append((parent_key, nested_dict))
    else:
        return nested_dict
 
    return {tuple(k): v for k, v in items}

def _unflatten(ary, space, nested_dict=None, idx=0):
    outer_call = False
    if nested_dict is None:
        outer_call = True
        nested_dict = {}

    for k, v in space.items():
        if isinstance(v, MutableMapping) or isinstance(v, gym.spaces.Dict):
            nested_dict[k] = {}
            _, idx = _unflatten(ary, v, nested_dict[k], idx)
        else:
            nested_dict[k] = ary[idx]
            idx += 1

    if outer_call:
        return nested_dict

    return nested_dict, idx

def pack_obs_space(obs_space, dtype=np.float32):
    assert(isinstance(obs_space, gym.Space)), 'Arg must be a gym space'

    if isinstance(obs_space, gym.spaces.Box):
        return obs_space

    flat = _flatten(obs_space)

    n = 0
    for e in flat.values():
        n += np.prod(e.shape)

    return gym.spaces.Box(
        low=-2**20, high=2**20,
        shape=(int(n),), dtype=dtype
    )

def pack_atn_space(atn_space):
    assert(isinstance(atn_space, gym.Space)), 'Arg must be a gym space'

    if isinstance(atn_space, gym.spaces.Discrete):
        return atn_space

    flat = _flatten(atn_space)

    lens = []
    for e in flat.values():
        lens.append(e.n)

    return gym.spaces.MultiDiscrete(lens) 

def flatten_ob(ob, dtype=None):
    flat = _flatten(ob)

    if type(ob) == np.ndarray:
        flat = {'': flat}

    vals = [e.ravel() for e in flat.values()]
    vals = np.concatenate(vals)

    if dtype is not None:
        vals = vals.astype(dtype)

    return vals

def pack_obs(obs, dtype=None):
    return {k: flatten_ob(v, dtype) for k, v in obs.items()}

def batch_obs(obs):
    return np.stack(list(obs.values()), axis=0)

def pack_and_batch_obs(obs):
    obs = pack_obs(obs)
    return batch_obs(obs)

def unpack_batched_obs(obs_space, packed_obs):
    assert(isinstance(obs_space, gym.Space)), 'First arg must be a gym space'

    batch = packed_obs.shape[0]
    obs = {}
    idx = 0

    flat_obs_space = _flatten(obs_space)
    for key_list, val in flat_obs_space.items():
        obs_ptr = obs
        for key in key_list[:-1]:
            if key not in obs_ptr:
                obs_ptr[key] = {}
            obs_ptr = obs_ptr[key]

        key = key_list[-1]
        inc = np.prod(val.shape)
        obs_ptr[key] = packed_obs[:, idx:idx + inc].reshape(batch, *val.shape)
        idx = idx + inc

    return obs