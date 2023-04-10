from pdb import set_trace as T
import itertools

import numpy as np
from contextlib import nullcontext
from collections import defaultdict
from collections import OrderedDict

import gym
from pettingzoo.utils.env import ParallelEnv

from pufferlib import utils


class Featurizer:
    def __init__(self, teams, team_id):
        self.teams = teams
        self.team_id = team_id

        assert type(teams) == dict
        assert team_id in teams

        self.num_teams = len(teams)
        self.team_size = len(teams[team_id])

    def reset(self):
        pass

    def step(self, team_obs):
        pass


class Binding:
    @property
    def raw_env_cls(self):
        '''Returns the original, unwrapped environment class used to create this binding, if available'''
        if self._raw_env_cls is None:
            raise ValueError('raw_env_cls not available when binding is not passed an env_cls')
        return self._raw_env_cls

    def raw_env_creator(self):
        '''Returns the original, unwrapped env_creator function used to create this binding'''
        with utils.Suppress() if self._suppress_env_prints else nullcontext():
            if self._raw_env_cls is None:
                return self._raw_env_creator(*self._default_args, **self._default_kwargs)
            else:
                return self._raw_env_cls(*self._default_args, **self._default_kwargs)

    @property
    def env_cls(self):
        '''Returns wrapped PufferEnv class created by this binding'''
        return self._env_cls

    def env_creator(self):
        '''Returns the wrapped PufferEnv env_creator function created by this binding'''
        return self._env_cls(*self._default_args, **self._default_kwargs)

    @property
    def raw_single_observation_space(self):
        '''Returns the unwrapped, structured observation space of a single agent.
        
        PufferLib currently assumes that all agents share the same observation space'''
        return self._raw_single_observation_space

    @property
    def featurized_single_observation_space(self):
        '''Returns the wrapped, structured, featurized observation space of a single agent.
        
        PufferLib currently assumes that all agents share the same observation space'''
        return self._featurized_single_observation_space

    @property
    def single_observation_space(self):
        '''Returns the wrapped, flat observation space of a single agent.
        
        PufferLib currently assumes that all agents share the same observation space'''
        return self._single_observation_space

    @property
    def raw_single_action_space(self):
        '''Returns the unwrapped, structured action space of a single agent.
        
        PufferLib currently assumes that all agents share the same action space'''
        return self._raw_single_action_space

    @property
    def single_action_space(self):
        '''Returns the wrapped, flat action space of a single agent.
        
        PufferLib currently assumes that all agents share the same action space'''
        return self._single_action_space

    @property
    def max_agents(self):
        '''Returns the maximum number of agents in the environment'''
        return self._max_agents

    @property
    def env_name(self):
        '''Returns the environment name'''
        return self._env_name

    def __init__(self,
            env_cls=None, 
            env_creator=None,
            default_args=[],
            default_kwargs={},
            env_name=None,
            featurizer_cls=None,
            featurizer_args=[],
            featurizer_kwargs={},
            reward_shaper=None,
            teams=None,
            emulate_flat_obs=True,
            emulate_flat_atn=True,
            emulate_const_horizon=None,
            emulate_const_num_agents=True,
            suppress_env_prints=True,
            record_episode_statistics=True,
            obs_dtype=np.float32):
        '''PufferLib's core Binding class.
        
        Wraps the provided Gym or PettingZoo environment in a PufferEnv that
        behaves like a normal PettingZoo environment with several simplifications:
            - The observation space is flattened to a single vector
            - The action space is flattened to a single vector
            - The environment caches observation and action spaces for improved performance
            - The environment is reset to a fixed horizon
            - The environment is padded to a fixed number of agents in sorted order
            - If originally single-agent, the environment is wrapped in a PettingZoo environment
            - The environment records additional statistics
            - The environment has suppressed stdout and stderr to avoid poluting the console
            - The environment contains additional error checking

        The Binding class additionally provides utility functions for interacting with complex
        observation and action spaces.

        Args: 
            env_cls: Environment class to wrap. Specify this or env_creator
            env_creator: Environment creation function to wrap. Specify this or env_cls
            default_args: Default arguments for binding.env_creator and binding.raw_env_creator
            default_kwargs: Default keyword arguments for binding.env_creator and binding.raw_env_creator
            env_name: Name of the environment
            featurizer_cls: Featureizer class to use
            reward_shaper: Reward shaper to use
            emulate_flat_obs: Whether the observation space requires flattening
            emulate_flat_atn: Whether the action space requires flattening
            emulate_const_horizon: Fixed max horizon for resets, None if not applicable
            emulate_const_num_agents: Whether to pad to len(env.possible_agents) observations
            suppress_env_prints: Whether to consume all environment prints
            record_episode_statistics: Whether to record additional episode statistics
            obs_dtype: Observation data type
        '''
        assert (env_cls is None) != (env_creator is None), \
            'Specify only one of env_cls (preferred) or env_creator'

        self._env_name = env_name
        self._default_args = default_args
        self._default_kwargs = default_kwargs

        self._raw_env_cls = env_cls
        self._raw_env_creator = env_creator
        self._suppress_env_prints = suppress_env_prints

        raw_local_env = self.raw_env_creator()

        # TODO: Consider integrating these?
        # env = wrappers.AssertOutOfBoundsWrapper(env)
        # env = wrappers.OrderEnforcingWrapper(env)

        class PufferEnv(ParallelEnv):
            @utils.profile
            def _create_env(self, *args, **kwargs):
                with utils.Suppress() if suppress_env_prints else nullcontext():
                    if env_cls is None:
                        return env_creator(*args, **kwargs)
                    else:
                        return env_cls(*args, **kwargs) 

            @utils.profile
            def _reset_env(self):
                with utils.Suppress() if suppress_env_prints else nullcontext():
                    return self.env.reset()

            @utils.profile
            def _step_env(self, actions):
                with utils.Suppress() if suppress_env_prints else nullcontext():
                    return self.env.step(actions)

            @utils.profile
            def __init__(self, *args, env=None, **kwargs):
                # Populated by utils.profile decorator
                self.timers = {}
                self.prestep_timer = utils.Profiler()
                self.poststep_timer = utils.Profiler()
                self.timers['prestep_timer'] = self.prestep_timer
                self.timers['poststep_timer'] = self.poststep_timer

                if env is None:
                    self.env = self._create_env(*args, **kwargs)
                else:
                    self.env = env

                self.dummy_obs = {}
                self._step = 0
                self.done = False

                assert obs_dtype in {np.float32, np.uint8}
                self.obs_dtype = obs_dtype
                if obs_dtype == np.uint8:
                    self._obs_min = np.iinfo(obs_dtype).min
                    self._obs_max = np.iinfo(obs_dtype).max
                elif obs_dtype == np.float32:
                    self._obs_min = np.finfo(obs_dtype).min
                    self._obs_max = np.finfo(obs_dtype).max
 
                self.emulate_flat_obs = emulate_flat_obs
                self.emulate_flat_atn = emulate_flat_atn
                self.emulate_const_horizon = emulate_const_horizon
                self.emulate_const_num_agents = emulate_const_num_agents
                self.emulate_multiagent = not utils.is_multiagent(self.env)
                self.suppress_env_prints = suppress_env_prints
                self.record_episode_statistics = record_episode_statistics

                if self.emulate_multiagent:
                    assert teams is None, 'Single agent env cannot specify teams'
                    self._teams = {1: [1]}

                # Standardize property vs method obs/atn space interface
                if self.emulate_multiagent:
                    self.possible_agents = [1]
                else:
                    self.possible_agents = self.env.possible_agents

                # Assign teams if not provided
                if teams is None:
                    self._teams = {a:[a] for a in self.possible_agents}
                else:
                    team_agents = set(itertools.chain.from_iterable(teams.values()))
                    assert set(self.possible_agents) == set(team_agents)
                    self._teams = teams

                # Initialize feature parser and reward shaper
                self.featurizers = {
                    team_id: featurizer_cls(
                        self._teams, team_id, *featurizer_args, **featurizer_kwargs)
                    for team_id, team in self._teams.items()
                }
                self.reward_shaper = reward_shaper

                # Override possible agents if teams are provided
                if teams is not None:
                    self.possible_agents = list(teams.keys())

                # Manual LRU since functools.lru_cache is not pickleable
                self.observation_space_cache = {}
                self.action_space_cache = {}

                # Cache observation and action spaces
                if self.emulate_const_num_agents:
                    for agent in self.possible_agents:
                        self.observation_space(agent)
                        self.action_space(agent)

                # Set env metadata
                if hasattr(self.env, 'metadata'):
                    self.metadata = self.env.metadata
                else:
                    self.metadata = {}

            @property
            def max_agents(self):
                return len(self.possible_agents)

            @utils.profile
            def action_space(self, agent):
                '''Flattened (MultiDiscrete) and cached action space'''

                if agent in self.action_space_cache:
                    return self.action_space_cache[agent]

                # Get single/multiagent action space
                if self.emulate_multiagent:
                    atn_space = self.env.action_space
                elif teams is not None:
                    atn_space = gym.spaces.Dict(
                        {a: self.env.action_space(a) for a in teams[agent]}
                    )
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

            @utils.profile
            def observation_space(self, team_id: int):
                '''Flattened (Box) and cached observation space'''
                if team_id in self.observation_space_cache:
                    return self.observation_space_cache[team_id]

                # Get single/multiagent observation space
                if self.emulate_multiagent:
                    obs_space = self.env.observation_space
                elif teams is not None:
                    obs_space = gym.spaces.Dict({
                        a: self.env.observation_space(a) for a in teams[team_id]}
                    )
                else:
                    obs_space = self.env.observation_space(team_id)

                dummy = _zero(obs_space.sample())

                # Initialize obs with dummy featurizer
                if self.featurizers:
                    dummy_featurizer = featurizer_cls(
                        self._teams, team_id, *featurizer_args, **featurizer_kwargs
                    )
                    dummy_featurizer.reset(dummy)
                    dummy = dummy_featurizer(dummy, self._step)
                    self._featurized_single_observation_space = _make_space_like(dummy)

                if self.emulate_flat_obs:
                    dummy = _flatten_ob(dummy, self.obs_dtype)

                self.dummy_obs[team_id] = dummy

                obs_space = gym.spaces.Box(
                    low=self._obs_min, high=self._obs_max,
                    shape=dummy.shape, dtype=self.obs_dtype
                )

                self.observation_space_cache[team_id] = obs_space
                return obs_space

            @utils.profile
            def _process_obs(self, obs, reset=False):
                '''Process observation. Shared by reset and step.'''
                team_obs = {}
                for team_id, team in self._teams.items():
                    team_obs[team_id] = {}
                    for agent_id in team:
                        if agent_id in obs:
                            team_obs[team_id][agent_id] = obs[agent_id]

                    this_team_obs = team_obs[team_id]

                    # Feature parser is stateful
                    if reset:
                        self.featurizers[team_id].reset(this_team_obs)

                    if len(this_team_obs) > 0:
                        team_obs[team_id] = self.featurizers[team_id](this_team_obs, self._step)
                    else:
                        del team_obs[team_id]

                if self.emulate_const_num_agents:
                    for k in self.dummy_obs:
                        if k not in team_obs:                                                  
                            team_obs[k] = self.dummy_obs[k]
            
                if self.emulate_flat_obs:
                    team_obs = _pack_obs(team_obs, self.obs_dtype)

                return team_obs

            def seed(self, seed):
                '''Seed the environment. Note that this is deprecated in new gym versions.'''
                self.env.seed(seed)

            @utils.profile
            def reset(self):
                '''Reset the environment and return observations'''
                self._epoch_returns = defaultdict(float)
                self._epoch_lengths = defaultdict(int)

                self.reset_calls_step = False
                obs = self._reset_env()

                if self.emulate_multiagent:
                    obs = {1: obs}
                    self.agents = [1]
                else:
                    self.agents = self.env.agents

                self.done = False

                # Some envs implement reset by calling step
                if not self.reset_calls_step:
                    obs = self._process_obs(obs, reset=True)

                # TODO: Figure out how to move featurizer to here

                self._step = 0
                return obs

            @utils.profile
            def step(self, team_actions, **kwargs):
                '''Step the environment and return (observations, rewards, dones, infos)'''
                assert not self.done, 'step after done'
                self.reset_calls_step = True

                # Action shape test
                if __debug__:
                    for agent, atns in team_actions.items():
                        assert self.action_space(agent).contains(atns)

                # Unpack actions from teams
                actions = {}
                for team_id, team in self._teams.items():
                    team_atns = np.split(team_actions[team_id], len(team))
                    for agent_id, atns in zip(team, team_atns):
                        actions[agent_id] = atns

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
                    if self.emulate_const_horizon is not None:
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

                    # Sort by possible_agents ordering
                    sorted_obs, sorted_rewards, sorted_dones, sorted_infos = {}, {}, {}, {}
                    for agent in self.possible_agents:
                        self._epoch_lengths[agent] += 1
                        self._epoch_returns[agent] += rewards[agent]

                        if self.record_episode_statistics and dones[agent]:
                            if 'episode' not in infos[agent]:
                                infos[agent]['episode'] = {}

                            infos[agent]['episode']['r'] = self._epoch_returns[agent]
                            infos[agent]['episode']['l'] = self._epoch_lengths[agent]

                            self._epoch_lengths[agent] = 0
                            self._epoch_returns[agent] = 0
 
                        sorted_obs[agent] = obs[agent]
                        sorted_rewards[agent] = rewards[agent]
                        sorted_dones[agent] = dones[agent]
                        sorted_infos[agent] = infos[agent]

                    obs, rewards, dones, infos = sorted_obs, sorted_rewards, sorted_dones, sorted_infos

                    # Observation shape test
                    if __debug__:
                        for agent, ob in obs.items():
                            if not self.observation_space(agent).contains(ob):
                                T()
                            assert self.observation_space(agent).contains(ob)

                return obs, rewards, dones, infos

        self._env_cls = PufferEnv
        local_env = PufferEnv(env=raw_local_env)

        self._default_agent = local_env.possible_agents[0]
        self._max_agents = local_env.max_agents
        self._emulate_multiagent = local_env.emulate_multiagent

        self._single_observation_space = local_env.observation_space(self._default_agent)
        self._featurized_single_observation_space = local_env._featurized_single_observation_space
        self._single_action_space = local_env.action_space(self._default_agent)

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

def _make_space_like(nested_dict):
    types = (OrderedDict, list, dict)

    gym_space = {}
    for k, v in nested_dict.items():
        if type(v) in types:
            gym_space[k] = _make_space_like(v)
        elif type(v) == np.ndarray:
            # TODO: Set min/max by dtype
            gym_space[k] = gym.spaces.Box(
                low=-2**20, high=2**20,
                shape=v.shape, dtype=v.dtype
            )
        else:
            assert False, f'Invalid type for featurized obs: {type(v)}'

    return gym.spaces.Dict(gym_space)

def _flatten(nested_dict):
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

    # TODO: Find a way to flip the check and the loop
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