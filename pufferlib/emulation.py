from pdb import set_trace as T

import numpy as np
import time
import functools
from contextlib import nullcontext

from collections import OrderedDict
from collections.abc import MutableMapping
import inspect

import gym
from pettingzoo.utils.env import ParallelEnv

from pufferlib import utils

import time


class Binding:
    @property
    def raw_env_cls(self):
        if self._raw_env_cls is None:
            raise ValueError('raw_env_cls not available when binding is not passed an env_cls')
        return self._raw_env_cls

    def raw_env_creator(self):
        if self._raw_env_cls is None:
            return self._raw_env_creator(*self._default_args, **self._default_kwargs)
        else:
            return self._raw_env_cls(*self._default_args, **self._default_kwargs)

    @property
    def env_cls(self):
        return self._env_cls

    def env_creator(self):
        return self._env_cls(*self._default_args, **self._default_kwargs)

    @property
    def single_observation_space(self):
        return self._single_observation_space

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def raw_single_observation_space(self):
        return self._raw_single_observation_space

    @property
    def raw_single_action_space(self):
        return self._raw_single_action_space

    @property
    def default_agent(self):
        return self._default_agent

    @property
    def env_name(self):
        return self._env_name

    def __init__(self,
            env_cls=None, 
            env_creator=None,
            default_args=[],
            default_kwargs={},
            env_name=None,
            feature_parser=None,
            reward_shaper=None,
            env_includes_reset=False,
            emulate_flat_obs=True,
            emulate_flat_atn=True,
            emulate_const_horizon=1024,
            emulate_const_num_agents=True,
            suppress_env_prints=False,
            obs_dtype=np.float32):
        '''Base class for PufferLib bindings

        Args: 
            env_name: Name of the environment
            env_cls: Environment class to wrap
            env_args: Arguments for env_cls
            env_kwargs: Keyword arguments for env_cls
        '''
        assert (env_cls is None) != (env_creator is None), \
            'Specify only one of env_cls (preferred) or env_creator'

        self._env_name = env_name
        self._default_args = default_args
        self._default_kwargs = default_kwargs

        self._raw_env_cls = env_cls
        self._raw_env_creator = env_creator

        raw_local_env = self.raw_env_creator()

        # Consider integrating these?
        #env = wrappers.AssertOutOfBoundsWrapper(env)
        #env = wrappers.OrderEnforcingWrapper(env)
        class PufferEnv(ParallelEnv):
            @utils.profile
            def _create_env(self, *args, **kwargs):
                if env_cls is None:
                    return env_creator(*args, **kwargs)
                else:
                    return env_cls(*args, **kwargs) 

            @utils.profile
            def _reset_env(self):
                return self.env.reset()

            @utils.profile
            def _step_env(self, actions):
                return self.env.step(actions)

            @utils.profile
            def __init__(self, *args, env=None, **kwargs):
                # Populated by utils.profile decorator
                self.timers = {}
                self.prestep_timer = utils.Profiler()
                self.poststep_timer = utils.Profiler()
                self.timers['prestep_timer'] = self.prestep_timer
                self.timers['poststep_timer'] = self.poststep_timer

                # Infer obs space from first agent
                # Assumes all agents have the same obs space
                with utils.Suppress() if suppress_env_prints else nullcontext():
                    if env is None:
                        self.env = self._create_env(*args, **kwargs)
                    else:
                        self.env = env

                self.dummy_obs = {}
                self._step = 0
                self.done = False
                self.obs_dtype = obs_dtype

                self.feature_parser = feature_parser
                self.reward_shaper = reward_shaper
                self.env_includes_reset = env_includes_reset

                self.emulate_flat_obs = emulate_flat_obs
                self.emulate_flat_atn = emulate_flat_atn
                self.emulate_const_horizon = emulate_const_horizon
                self.emulate_const_num_agents = emulate_const_num_agents
                self.emulate_multiagent = not utils.is_multiagent(self.env)
                self.suppress_env_prints = suppress_env_prints

                # Manual LRU since functools.lru_cache is not pickleable
                self.observation_space_cache = {}
                self.action_space_cache = {}

                # Standardize property vs method obs/atn space interface
                if self.emulate_multiagent:
                    self.possible_agents = [1]
                else:
                    self.possible_agents = self.env.possible_agents

                # Set env metadata
                if hasattr(self.env, 'metadata'):
                    self.metadata = self.env.metadata
                else:
                    self.metadata = {}

            def num_agents(self):
                return len(self.possible_agents)

            @utils.profile
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

                if agent in self.action_space_cache:
                    return self.action_space_cache[agent]

                # Get single/multiagent action space
                if self.emulate_multiagent:
                    atn_space = self.env.action_space
                else:
                    atn_space = self.env.action_space(agent)

                if self.emulate_flat_atn:
                    assert type(atn_space) in (gym.spaces.Dict, gym.spaces.Discrete, gym.spaces.MultiDiscrete)
                    if type(atn_space) == gym.spaces.Dict:
                        atn_space = _pack_atn_space(atn_space)
                    elif type(atn_space) == gym.spaces.Discrete:
                        atn_space = gym.spaces.MultiDiscrete([atn_space.n])
                    
                self.action_space_cache[agent] = atn_space

                return atn_space

            def structured_observation_space(self, agent: int):
                if self.feature_parser:
                    return self.feature_parser.spec

                if self.emulate_multiagent:
                    return self.env.observation_space
                else:
                    return self.env.observation_space(agent)

            @utils.profile
            def observation_space(self, agent: int):
                if agent in self.observation_space_cache:
                    return self.observation_space_cache[agent]

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
                    dummy = _flatten_ob(dummy, self.obs_dtype)

                obs_space = gym.spaces.Box(
                    low=-2**20, high=2**20,
                    shape=dummy.shape, dtype=self.obs_dtype
                )

                self.observation_space_cache[agent] = obs_space
                return obs_space

            @utils.profile
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
                    obs = _pack_obs(obs, self.obs_dtype)

                return obs

            # Deprecated Pettingzoo API
            def seed(self, seed):
                self.env.seed(seed)

            @utils.profile
            def reset(self):
                self.reset_calls_step = False

                obs = self._reset_env()

                if self.emulate_multiagent:
                    obs = {1: obs}
                    self.agents = [1]
                else:
                    self.agents = self.env.agents

                self.done = False

                # Populate dummy ob
                if self.emulate_const_num_agents:
                    for agent in self.possible_agents:
                        self.observation_space(agent)

                # Some envs implement reset by calling step
                if not self.reset_calls_step:
                    obs = self._process_obs(obs)

                self._step = 0
                return obs

            @utils.profile
            def step(self, actions, **kwargs):
                assert not self.done, 'step after done'
                self.reset_calls_step = True

                # Action shape test
                if __debug__:
                    for agent, atns in actions.items():
                        assert self.action_space(agent).contains(atns)

                # Unpack actions
                with self.prestep_timer:
                    if self.emulate_flat_atn:
                        for k in list(actions):
                            if k not in self.agents:
                                del(actions[k])
                                continue

                            v = actions[k]
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

                    ob, reward, done, info = self._step_env(action)

                    obs = {1: ob}
                    rewards = {1: reward}
                    dones = {1: done}
                    infos = {1: info}

                    if done:
                        self.done = True
                        self.agents = []

                else:
                    obs, rewards, dones, infos = self._step_env(actions)
                    self.agents = self.env.agents
                    self.done = len(self.agents) == 0

                # RLlib compat 
                assert '__all__' not in dones, 'Base env should not return __all__'

                self._step += 1
            
                obs = self._process_obs(obs)

                with self.poststep_timer:
                    if self.reward_shaper:
                        rewards = self.reward_shaper(rewards, self._step)

                    # Terminate episode at horizon or if all agents are done
                    if self.emulate_const_horizon:
                        assert self._step <= self.emulate_const_horizon
                        if self._step == self.emulate_const_horizon:
                            self.done = True

                    # Computed before padding dones. False if no agents
                    # Pad rewards/dones/infos
                    if self.emulate_const_num_agents:
                        for k in self.dummy_obs:
                            # TODO: Check that all keys are present
                            if k not in rewards:
                                rewards[k] = 0
                            if k not in infos:
                                infos[k] = {}
                            if k not in dones:
                                dones[k] = self.done

                    # Env wrapper already resets itself
                    #if self.env_includes_reset:
                    #    self.done = False

                    # Sort by possible_agents ordering
                    sorted_obs, sorted_rewards, sorted_dones, sorted_infos = {}, {}, {}, {}
                    for agent in self.possible_agents:
                        sorted_obs[agent] = obs[agent]
                        sorted_rewards[agent] = rewards[agent]
                        sorted_dones[agent] = dones[agent]
                        sorted_infos[agent] = infos[agent]

                    obs, rewards, dones, infos = sorted_obs, sorted_rewards, sorted_dones, sorted_infos

                    # Observation shape test
                    if __debug__:
                        for agent, ob in obs.items():
                            assert self.observation_space(agent).contains(ob)

                return obs, rewards, dones, infos

        self._env_cls = PufferEnv
        local_env = PufferEnv(env=raw_local_env)

        self._default_agent = local_env.possible_agents[0]
        self._single_observation_space = local_env.observation_space(self._default_agent)
        self._single_action_space = local_env.action_space(self._default_agent)
        self._emulate_multiagent = local_env.emulate_multiagent

        if self._emulate_multiagent:
            self._raw_single_observation_space = raw_local_env.observation_space
            self._raw_single_action_space = raw_local_env.action_space
        else:
            self._raw_single_observation_space = raw_local_env.observation_space(self._default_agent)
            self._raw_single_action_space = raw_local_env.action_space(self._default_agent)



def unpack_batched_obs(obs_space, packed_obs):
    '''Unpack a batch of observations into the original observation space
    
    Call this funtion in the forward pass of your network
    '''

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

def _zero(ob):
    if type(ob) == np.ndarray:
        ob.fill(0)
    elif type(ob) in (dict, OrderedDict):
        for k, v in ob.items():
            _zero(ob[k])
    else:
        for v in ob:
            _zero(v)
    return ob

def _flatten(nested_dict, parent_key=None):
    types = (gym.spaces.Dict, OrderedDict, list, dict, tuple)

    if type(nested_dict) not in types:
        return nested_dict

    stack = [((), nested_dict)]
    flat_dict = {}
    while stack:
        path, current = stack.pop()
        for k, v in current.items():
            new_key = path + (k,)
            if type(v) in types:
                stack.append((new_key, v))
            else:
                flat_dict[new_key] = v

    return flat_dict

def _unflatten(ary, space, nested_dict=None, idx=0):
    outer_call = False
    if nested_dict is None:
        outer_call = True
        nested_dict = {}

    #TODO: Find a way to flip the check and the loop
    # (Added for Gym microrts)
    if type(space)  == gym.spaces.MultiDiscrete:
        return ary

    types = (gym.spaces.Dict, OrderedDict, list, dict, tuple)
    for k, v in space.items():
        if type(v) in types:
            nested_dict[k] = {}
            _, idx = _unflatten(ary, v, nested_dict[k], idx)
        else:
            nested_dict[k] = ary[idx]
            idx += 1

    if outer_call:
        return nested_dict

    return nested_dict, idx

def _pack_obs_space(obs_space, dtype=np.float32):
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

def _pack_atn_space(atn_space):
    assert(isinstance(atn_space, gym.Space)), 'Arg must be a gym space'

    if isinstance(atn_space, gym.spaces.Discrete):
        return atn_space

    flat = _flatten(atn_space)

    lens = []
    for e in flat.values():
        lens.append(e.n)

    return gym.spaces.MultiDiscrete(lens) 

def _flatten_ob(ob, dtype=None):
    # TODO: Find a better way to handle Atari
    if type(ob) == gym.wrappers.frame_stack.LazyFrames:
       ob = np.array(ob)

    #assert type(ob) == np.array

    flat = _flatten(ob)

    if type(ob) == np.ndarray:
        flat = {'': flat}

    # Preallocate the memory for the concatenated tensor
    tensors = flat.values()

    if dtype is None:
        tensors = list(tensors)
        dtype = tensors[0].dtype

    tensor_sizes = [tensor.size for tensor in tensors] 
    prealloc = np.empty(sum(tensor_sizes), dtype=dtype)

    # Fill the concatenated tensor with the flattened tensors
    start = 0
    for tensor, size in zip(tensors, tensor_sizes):
        end = start + size
        prealloc[start:end] = tensor.ravel()
        start = end

    return prealloc

def _pack_obs(obs, dtype=None):
    return {k: _flatten_ob(v, dtype) for k, v in obs.items()}

def _batch_obs(obs):
    return np.stack(list(obs.values()), axis=0)

def _pack_and_batch_obs(obs):
    obs = _pack_obs(obs)
    return _batch_obs(obs)