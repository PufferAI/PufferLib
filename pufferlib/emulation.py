from pdb import set_trace as T
import itertools
import numpy as np
from contextlib import nullcontext
from collections import defaultdict, OrderedDict

import gym
from pettingzoo.utils.env import ParallelEnv
from pufferlib import utils

def group_team_obs(obs, teams):
    team_obs = {}
    for team_id, team in teams.items():
        team_obs[team_id] = {}
        for agent_id in team:
            if agent_id in obs:
                team_obs[team_id][agent_id] = obs[agent_id]
    return team_obs


class Featurizer:
    def __init__(self, teams, team_id):
        self.teams = teams
        self.team_id = team_id

        assert isinstance(teams, dict)
        assert team_id in teams

        self.num_teams = len(teams)
        self.team_size = len(teams[team_id])

    def reset(self, team_obs):
        return

    def __call__(self, team_obs, step):
        key = list(team_obs.keys())[0]
        return team_obs[key]


class GymToPettingZooParallelWrapper(ParallelEnv):
    def __init__(self, env: gym.Env):
        self.env = env
        self.possible_agents = [1]

    def reset(self):
        self.agents = [1]
        return {1: self.env.reset()}

    def step(self, actions):
        ob, reward, done, info = self.env.step(actions[1])

        if done:
            self.agents = []

        return {1: ob}, {1: reward}, {1: done}, {1: info}

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

# TODO: Consider integrating these?
# env = wrappers.AssertOutOfBoundsWrapper(env)
# env = wrappers.OrderEnforcingWrapper(env)
def make_puffer_env_cls(scope, raw_obs):
    class PufferEnv(ParallelEnv):
        def _initialize_timers(self):
            # Populated by utils.profile decorator
            self.timers = {}
            self.prestep_timer = utils.Profiler()
            self.poststep_timer = utils.Profiler()
            self.featurizer_timer = utils.Profiler()
            self.timers['prestep_timer'] = self.prestep_timer
            self.timers['poststep_timer'] = self.poststep_timer
            self.timers['featurizer_timer'] = self.featurizer_timer

        @utils.profile
        def __init__(self, *args, env=None, **kwargs):
            self._initialize_timers()

            self.env = self._create_env(env, *args, **kwargs)
            self.pz_env = self.env if utils.is_multiagent(self.env) else GymToPettingZooParallelWrapper(self.env)

            self._step = 0
            self.done = False
            self._obs_min, self._obs_max = utils._get_dtype_bounds(scope.obs_dtype)

            # Assign valid teams. Autogen 1 team per agent if not provided
            self._teams = {a:[a] for a in self.env.possible_agents} if scope.teams is None else scope.teams
            assert set(self.env.possible_agents) == {item for team in self._teams.values() for item in team}
            self.possible_agents = list(self._teams.keys())

            # Initialize feature parser and reward shaper
            self.featurizers = {
                team_id: scope.featurizer_cls(
                    self._teams, team_id, *scope.featurizer_args, **scope.featurizer_kwargs)
                for team_id in self._teams
            }
 
            # TODO v0.4: Implement reward shaper API
            self.reward_shaper = scope.reward_shaper

            # Manual LRU since functools.lru_cache is not pickleable
            self._featurized_single_observation_space = None
            self._single_action_space = None

            self.obs_space_cache, self.atn_space_cache = {}, {}
            self.obs_space_cache = {team_id: self.observation_space(team_id) for team_id in self.possible_agents}
            self.atn_space_cache = {team_id: self.action_space(team_id) for team_id in self.possible_agents}

            # Set env metadata
            self.metadata = self.env.metadata if hasattr(self.env, 'metadata') else {}

        @utils.profile
        def _create_env(self, env, *args, **kwargs):
            if env is not None:
                return env

            with utils.Suppress() if scope.suppress_env_prints else nullcontext():
                if scope.env_cls:
                    return scope.env_cls(*args, **kwargs)
                return scope.env_creator(*args, **kwargs)
        
        @utils.profile
        def _reset_env(self, seed):
            with utils.Suppress() if scope.suppress_env_prints else nullcontext():
                # Handle seeding with different gym versions
                try:
                    return self.env.reset(seed=seed)
                except:
                    self.env.seed(seed)
                    return self.env.reset()

        @utils.profile
        def _step_env(self, actions):
            with utils.Suppress() if scope.suppress_env_prints else nullcontext():
                return self.env.step(actions)

        @property
        def max_agents(self):
            return len(self.possible_agents)

        @utils.profile
        def observation_space(self, agent):
            #Flattened (Box) and cached observation space
            if agent in self.obs_space_cache:
                return self.obs_space_cache[agent]

            # Get single/multiagent observation space
            obs_space = {a: self.env.observation_space(a) for a in self.possible_agents} 

            # Initialize obs with dummy featurizer
            featurizer = scope.featurizer_cls(
                self._teams, agent, *scope.featurizer_args, **scope.featurizer_kwargs
            )

            team_obs = group_team_obs(raw_obs, self._teams)[agent]
            featurizer.reset(team_obs)
            obs = featurizer(team_obs, self._step)

            if self._featurized_single_observation_space is None:
                self._featurized_single_observation_space = _flatten_space(_make_space_like(obs))

            if scope.emulate_flat_obs:
                obs = _flatten_ob(obs, self._featurized_single_observation_space, scope.obs_dtype)

            self.pad_obs = 0 * obs

            obs_space = gym.spaces.Box(
                low=self._obs_min, high=self._obs_max,
                shape=obs.shape, dtype=scope.obs_dtype
            )

            return obs_space

        @utils.profile
        def action_space(self, agent):
            #Flattened (MultiDiscrete) and cached action space
            if agent in self.atn_space_cache:
                return self.atn_space_cache[agent]
                
            if self._single_action_space is None:
                self._single_action_space = _flatten_space(self.env.action_space(agent))

            atn_space = gym.spaces.Dict({a: self.env.action_space(a) for a in self._teams[agent]})

            if scope.emulate_flat_atn:
                assert type(atn_space) in (gym.spaces.Dict, gym.spaces.Discrete, gym.spaces.MultiDiscrete)
                if type(atn_space) == gym.spaces.Dict:
                    atn_space = _pack_atn_space(atn_space)
                elif type(atn_space) == gym.spaces.Discrete:
                    atn_space = gym.spaces.MultiDiscrete([atn_space.n])

            return atn_space

        @utils.profile
        def reset(self, seed=None):
            #Reset the environment and return observations
            self._epoch_returns = defaultdict(float)
            self._epoch_lengths = defaultdict(int)

            obs = group_team_obs(self._reset_env(seed), self._teams)
            for team_id, team_obs in obs.items():
                self.featurizers[team_id].reset(team_obs)
                obs[team_id] = self.featurizers[team_id](team_obs, self._step)

            self.agents = self.env.agents
            self.done = False

            # Sort observations according to possible_agents
            # All keys must be present in possible_agents on reset
            obs = {a: obs[a] for a in self.possible_agents}

            orig_obs = obs
            if scope.emulate_flat_obs:
                obs = _pack_obs(obs, self._featurized_single_observation_space, scope.obs_dtype)

            if __debug__:
                for agent, ob in obs.items():
                    assert self.observation_space(agent).contains(ob)

                packed_obs = np.stack(obs.values())
                unpacked = unpack_batched_obs(self._featurized_single_observation_space, packed_obs)
                ret = _compare_observations(orig_obs, unpacked)
                assert ret, 'Observation packing/unpacking mismatch'

            self._step = 0
            return obs

        @utils.profile
        def step(self, team_actions, **kwargs):
            #Step the environment and return (observations, rewards, dones, infos)
            assert not self.done, 'step after done'

            # Action shape test
            if __debug__:
                for agent, atns in team_actions.items():
                    if not self.action_space(agent).contains(atns):
                        print(agent, atns)
                        T()
                    assert self.action_space(agent).contains(atns)

            # Unpack actions from teams
            actions = {}
            for team_id, team in self._teams.items():
                team_atns = np.split(team_actions[team_id], len(team))
                for agent_id, atns in zip(team, team_atns):
                    actions[agent_id] = atns

            # Unpack actions
            with self.prestep_timer:
                if scope.emulate_flat_atn:
                    for k in list(actions):
                        if k not in self.agents:
                            del(actions[k])
                            continue

                        actions[k] = _unflatten(actions[k], self._single_action_space)

            obs, rewards, dones, infos = self._step_env(actions)
            self.agents = self.env.agents
            self.done = len(self.agents) == 0
            self._step += 1

            with self.featurizer_timer:
                # Featurize observations for teams with at least 1 living agent
                obs = {
                    team_id: self.featurizers[team_id](team_obs, self._step)
                    for team_id, team_obs in group_team_obs(obs, self._teams).items() if team_obs
                }

            with self.poststep_timer:
                if self.reward_shaper:
                    rewards = self.reward_shaper(rewards, self._step)

                # Terminate episode at horizon or if all agents are done
                if scope.emulate_const_horizon is not None:
                    assert self._step <= scope.emulate_const_horizon
                    if self._step == scope.emulate_const_horizon:
                        self.done = True

                if scope.emulate_flat_obs:
                    obs = _pack_obs(obs, self._featurized_single_observation_space, scope.obs_dtype)

                # Computed before padding dones. False if no agents
                # Pad rewards/dones/infos
                if scope.emulate_const_num_agents:
                    for team in self._teams:
                        if team not in obs:                                                  
                            obs[team] = self.pad_obs
                        if team not in rewards:
                            rewards[team] = 0
                        if team not in infos:
                            infos[team] = {}
                        if team not in dones:
                            dones[team] = self.done

                # Sort by possible_agents ordering
                sorted_obs, sorted_rewards, sorted_dones, sorted_infos = {}, {}, {}, {}
                for agent in self.possible_agents:
                    sorted_obs[agent] = obs[agent]
                    sorted_rewards[agent] = rewards[agent]
                    sorted_dones[agent] = dones[agent]
                    sorted_infos[agent] = infos[agent]

                obs, rewards, dones, infos = sorted_obs, sorted_rewards, sorted_dones, sorted_infos

                # Record episode statistics
                for agent in self.possible_agents:
                    self._epoch_lengths[agent] += 1 
                    self._epoch_returns[agent] += rewards[agent]

                    if scope.record_episode_statistics and dones[agent]:
                        if 'episode' not in infos[agent]:
                            infos[agent]['episode'] = {}

                        infos[agent]['episode']['r'] = self._epoch_returns[agent]
                        infos[agent]['episode']['l'] = self._epoch_lengths[agent]

                        self._epoch_lengths[agent] = 0
                        self._epoch_returns[agent] = 0

                # Observation shape test
                if __debug__:
                    for agent, ob in obs.items():
                        assert self.observation_space(agent).contains(ob)

            return obs, rewards, dones, infos

    return PufferEnv


class Binding:
    def __init__(self,
            env_cls=None, 
            env_creator=None,
            default_args=[],
            default_kwargs={},
            env_name=None,
            featurizer_cls=Featurizer,
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

        scope = utils.dotdict(locals())
        del scope['self']

        raw_local_env = self.raw_env_creator()

        try:
            raw_local_env.seed(42)
            old_seed=True
        except:
            old_seed=False

        if old_seed:
            raw_obs = raw_local_env.reset()
        else:
            raw_obs = raw_local_env.reset(seed=42)

        self._env_cls = make_puffer_env_cls(scope, raw_obs=raw_obs)

        local_env = self._env_cls(scope, env=raw_local_env)

        self._default_agent = local_env.possible_agents[0]
        self._max_agents = local_env.max_agents

        self._single_observation_space = local_env.observation_space(self._default_agent)
        self._featurized_single_observation_space = local_env._featurized_single_observation_space
        self._single_action_space = local_env.action_space(self._default_agent)

        self._raw_single_observation_space = raw_local_env.observation_space(self._default_agent)
        self._raw_single_action_space = raw_local_env.action_space(self._default_agent)

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


def unpack_batched_obs(flat_space, packed_obs):
    '''Unpack a batch of observations into the original observation space
    
    Call this funtion in the forward pass of your network
    '''

    if not isinstance(flat_space, dict):
        return packed_obs.reshape(packed_obs.shape[0], *flat_space.shape)

    batch = packed_obs.shape[0]
    batched_obs = {}

    idx = 0
    for key_list, space in flat_space.items():
        current_dict = batched_obs
        inc = np.prod(space.shape)

        for key in key_list[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

        last_key = key_list[-1]
        current_dict[last_key] = packed_obs[:, idx:idx + inc].reshape(batch, *space.shape)
        idx += inc

    return batched_obs

def _compare_observations(obs, batched_obs):
    def _compare_arrays(array1, array2):
        return np.allclose(array1, array2)

    def _compare_helper(obs, batched_obs, agent_idx):
        if isinstance(batched_obs, np.ndarray):
            return _compare_arrays(obs, batched_obs[agent_idx])
        elif isinstance(obs, (dict, OrderedDict)):
            for key in obs:
                if not _compare_helper(obs[key], batched_obs[key], agent_idx):
                    return False
            return True
        elif isinstance(obs, (list, tuple)):
            for idx, elem in enumerate(obs):
                if not _compare_helper(elem, batched_obs[idx], agent_idx):
                    return False
            return True
        else:
            raise ValueError(f"Unsupported type: {type(obs)}")

    if isinstance(batched_obs, dict):
        agent_indices = range(len(next(iter(batched_obs.values()))))
    else:
        agent_indices = range(batched_obs.shape[0])

    for agent_key, agent_idx in zip(sorted(obs.keys()), agent_indices):
        if not _compare_helper(obs[agent_key], batched_obs, agent_idx):
            return False

    return True

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

def _make_space_like(ob):
    if type(ob) == np.ndarray:
        # TODO: Set min/max by dtype
        return gym.spaces.Box(
            low=-2**20, high=2**20,
            shape=ob.shape, dtype=ob.dtype
        )

    # TODO: Handle Discrete (how to get max?)
    
    if type(ob) in (tuple, list):
        return gym.spaces.Tuple([_make_space_like(v) for v in ob])
 
    if type(ob) in (dict, OrderedDict):
        return gym.spaces.Dict({k: _make_space_like(v) for k, v in ob.items()})

    raise ValueError(f'Invalid type for featurized obs: {type(ob)}')

def _flatten_space(space):
    if isinstance(space, (gym.spaces.Box, gym.spaces.Discrete)):
        return space

    flat_keys = {}

    def _recursion_helper(current_space, key_list):
        if isinstance(current_space, (list, tuple, gym.spaces.Tuple)):
            for idx, elem in enumerate(current_space):
                new_key_list = key_list + (idx,)
                _recursion_helper(elem, new_key_list)
        elif isinstance(current_space, (dict, OrderedDict, gym.spaces.Dict)):
            for key, value in current_space.items():
                new_key_list = key_list + (key,)
                _recursion_helper(value, new_key_list)
        else:
            flat_keys[key_list] = current_space

    _recursion_helper(space, ())
    return flat_keys

def _flatten(nested, flat_space):
    if not isinstance(flat_space, dict):
        return nested.ravel()

    flat_data = []

    for key_list, space in flat_space.items():
        value = nested
        for key in key_list:
            value = value[key]
        flat_data.append(value.ravel())

    return flat_data

def _unflatten(flat, flat_space):
    if not isinstance(flat_space, dict):
        return flat.reshape(flat_space.shape)

    idx = 0
    nested_data = {}

    for key_list, space in flat_space.items():
        current_dict = nested_data
        inc = int(np.prod(space.shape))

        for key in key_list[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

        last_key = key_list[-1]

        if space.shape:
            current_dict[last_key] = flat[idx:idx + inc].reshape(space.shape)
        else:
            assert inc == 1
            current_dict[last_key] = flat[idx]

        idx += inc

    return nested_data

def _pack_obs_space(obs_space, dtype=np.float32):
    assert(isinstance(obs_space, gym.Space)), 'Arg must be a gym space'

    if isinstance(obs_space, gym.spaces.Box):
        return obs_space

    flat = _flatten_space(obs_space)

    n = 0
    for e in flat.values():
        n += np.prod(e.shape)

    return gym.spaces.Box(
        low=-2**20, high=3**20,
        shape=(int(n),), dtype=dtype
    )

def _pack_atn_space(atn_space):
    assert(isinstance(atn_space, gym.Space)), 'Arg must be a gym space'

    if isinstance(atn_space, gym.spaces.Discrete):
        return atn_space

    flat = _flatten_space(atn_space)

    lens = []
    for e in flat.values():
        if isinstance(e, gym.spaces.Discrete):
            lens.append(e.n)
        elif isinstance(e, gym.spaces.MultiDiscrete):
            lens += e.nvec.tolist()
        else:
            raise ValueError(f'Invalid action space: {e}')

    return gym.spaces.MultiDiscrete(lens) 

def _flatten_ob(ob, flat_space, dtype=None):
    # TODO: Find a better way to handle Atari
    if type(ob) == gym.wrappers.frame_stack.LazyFrames:
       ob = np.array(ob)

    tensors = _flatten(ob, flat_space)

    # Preallocate the memory for the concatenated tensor
    if type(tensors) == dict:
        tensors = tensors.values()

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

def _pack_obs(obs, flat_space, dtype=None):
    return {k: _flatten_ob(v, flat_space, dtype) for k, v in obs.items()}

def _batch_obs(obs):
    return np.stack(list(obs.values()), axis=0)

def _pack_and_batch_obs(obs):
    obs = _pack_obs(obs)
    return _batch_obs(obs)