from pdb import set_trace as T
import warnings
import numpy as np
from contextlib import nullcontext
try:
    from collections import defaultdict, OrderedDict, Mapping
except ImportError:
    # API changed in python 3.10
    from collections import defaultdict, OrderedDict
    from collections.abc import Mapping

import gym
from pettingzoo.utils.env import ParallelEnv

from pufferlib import utils
from pufferlib import exceptions


class Postprocessor:
    def __init__(self, env, teams, team_id):
        if not isinstance(teams, Mapping):
            raise ValueError(f'Teams is not a valid dict or mapping: {teams}')

        if team_id not in teams:
            raise ValueError(f'Team {team_id} not in teams {teams}')

        self.env = env
        self.teams = teams
        self.team_id = team_id

        self.num_teams = len(teams)
        self.team_size = len(teams[team_id])
        self.max_team_size = max([len(v) for v in self.teams.values()])

        self.dummy_ob = None

    def reset(self, team_obs):
        self.epoch_return = 0
        self.epoch_length = 0
        self.done = False

    def features(self, obs, step):
        '''Default featurizer pads observations to max team size'''

        if len(obs) == 0:
            #TODO: Re-enable this once nmmo is fixed
            team_id = self.team_id
            #raise ValueError('Observation is empty')
        else:
            team_id = [k for k, v in self.teams.items()
                if list(obs.keys())[0] in v][0]

        if self.dummy_ob is None:
            self.dummy_ob = utils.make_zeros_like(list(obs.values())[0])

        ret = []
        for agent in self.teams[team_id]:
            if agent in obs:
                ret.append(obs[agent])
            else:
                ret.append(self.dummy_ob)

        return ret

    def actions(self, actions, step):
        return actions

    def rewards(self, team_rewards, team_dones, team_infos, step):
        '''Default reward shaper sums team rewards'''
        return sum(team_rewards.values()), team_infos

    def infos(self, team_reward, team_done, team_infos, step):
        if self.done:
            return {}
        if team_done:
            team_infos['return'] = self.epoch_return
            team_infos['length'] = self.epoch_length
            self.done = True
        else:
            self.epoch_length += 1
            self.epoch_return += team_reward

        return team_infos

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

    def observation_space(self, agent):
        return self.env.observation_space

    def action_space(self, agent):
        return self.env.action_space

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

def make_puffer_env_cls(scope, raw_local_env, raw_obs):
    class PufferEnv(ParallelEnv):
        @utils.profile
        def __init__(self, *args, env=None, **kwargs):
            self.raw_env = self._create_env(env, *args, **kwargs)
            self.env = self.raw_env if utils.is_multiagent(self.raw_env) else GymToPettingZooParallelWrapper(self.raw_env)

            self.initialized = False
            self._step = 0

            self._obs_min, self._obs_max = utils._get_dtype_bounds(scope.obs_dtype)

            # Assign valid teams. Autogen 1 team per agent if not provided
            self._teams = {a:[a] for a in self.env.possible_agents} if scope.teams is None else scope.teams

            if set(self.env.possible_agents) != {item for team in self._teams.values() for item in team}:
                raise ValueError(f'Invalid teams: {self._teams} for possible_agents: {self.env.possible_agents}')

            self.possible_agents = list(self._teams.keys())
            self.default_team = self.possible_agents[0]

            # Initialize postprocessors
            self.postprocessors = {
                team_id: scope.postprocessor_cls(
                    self.env, self._teams, team_id,
                    *scope.postprocessor_args,
                    **scope.postprocessor_kwargs
                ) for team_id in self._teams
            }

            # Manual LRU since functools.lru_cache is not pickleable
            self._raw_observation_space = {}
            self._original_observation_space = {}
            self._flat_observation_space = {}
            self._flat_action_space = {}
            self._agent_action_space = {}

            self.obs_space_cache, self.atn_space_cache = {}, {}
            self.obs_space_cache = {team_id: self.observation_space(team_id) for team_id in self._teams}
            self.atn_space_cache = {team_id: self.action_space(team_id) for team_id in self._teams}

            # Set env metadata
            self.metadata = self.env.metadata if hasattr(self.env, 'metadata') else {}

        def _group_by_team(self, agent_data):
            team_data = {}
            for team_id, team in self._teams.items():
                team_data[team_id] = {}
                for agent_id in team:
                    if agent_id in agent_data:
                        team_data[team_id][agent_id] = agent_data[agent_id]
            return team_data

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
                return _seed_and_reset(self.env, seed)

        @utils.profile
        def _prestep(self, team_actions):
            # Agent key check
            if __debug__:
                for team, atns in team_actions.items():
                    if team not in self._teams:
                        raise exceptions.InvalidAgentError(team, self._teams)

            for team_id in self._teams:
                team_actions[team_id] = self._handle_actions(
                    team_actions[team_id], team_id)

            # Action shape test
            if __debug__:
                for team, atns in team_actions.items():
                    if not self.action_space(team).contains(atns):
                        raise ValueError(
                            f'action:\n{atns}\n for agent/team {team} not in '
                            f'action space:\n{self.action_space(team)}')

            # Unpack actions from teams
            actions = {}
            for team_id, team in self._teams.items():
                # TODO: Assert all keys present since actions are padded
                team_atns = np.split(team_actions[team_id], len(team))
                for agent_id, atns in zip(team, team_atns):
                    actions[agent_id] = atns

            # TODO: Do we want to support structured actions?
            if not scope.emulate_flat_atn:
                return actions

            # Unpack actions
            for k in list(actions):
                if k not in self.agents:
                    del(actions[k])
                    continue

                flat = actions[k]
                flat_space = self._agent_action_space[k]

                if not isinstance(flat_space, dict):
                    actions[k] = flat.reshape(flat_space.shape)
                elif () in flat_space:
                    actions[k] = flat[0]
                else:
                    nested_data = {}

                    for key_list, space in flat_space.items():
                        current_dict = nested_data

                        for key in key_list[:-1]:
                            if key not in current_dict:
                                current_dict[key] = {}
                            current_dict = current_dict[key]

                        last_key = key_list[-1]

                        if space.shape:
                            size = np.prod(space.shape)
                            current_dict[last_key] = flat[:size].reshape(space.shape)
                            flat = flat[size:]
                        else:
                            current_dict[last_key] = flat[0]
                            flat = flat[1:]

                    actions[k] = nested_data

            return actions

        @utils.profile
        def _step_env(self, actions):
            with utils.Suppress() if scope.suppress_env_prints else nullcontext():
                return self.env.step(actions)

        @utils.profile
        def _featurize(self, team_ob, team_id):
            # Featurize observations for teams with at least 1 living agent
            team_ob = self.postprocessors[team_id].features(team_ob, self._step)

            if __debug__:
                space = self._raw_observation_space[team_id]
                if not space.contains(team_ob):
                    raise ValueError(
                        f'Featurized observation:\n{team_ob}\n not in observation space:\n'
                        f'{space}\n for agent/team {team_id}. Common '
                        'causes include incorrect type (i.e. f32 vs f64)'
                    )

            return team_ob

        @utils.profile
        def _handle_actions(self, team_atns, team_id):
            return self.postprocessors[team_id].actions(team_atns, self._step)

        @utils.profile
        def _handle_infos(self, team_reward, team_done, team_infos, team_id):
            team_infos = self.postprocessors[team_id].infos(team_reward, team_done, team_infos, self._step)
            assert team_infos is not None, f'Postprocessor {team_id} returned None for infos'
            return team_infos

        @utils.profile
        def _shape_rewards(self, team_reward, team_dones, team_infos, team_id):
            # Shape rewards for teams with at least 1 living agent
            team_reward, team_infos = self.postprocessors[team_id].rewards(
                team_reward, team_dones, team_infos, self._step)

            if not isinstance(team_reward, (float, int)):
                raise ValueError(
                    f'Shaped reward {team_reward} of type '
                    f'{type(team_reward)} is not a float or int'
                )

            return team_reward, team_infos

        @utils.profile
        def _poststep(self, obs, rewards, dones, infos):
            # Group by teams
            obs = self._group_by_team(obs)
            rewards = self._group_by_team(rewards)
            dones = self._group_by_team(dones)

            # Featurize and shape rewards; pad data
            for team in self._teams:
                # TODO: Check if group_by should be adding in empty teams
                if obs[team]:
                    obs[team] = self._featurize(obs[team], team)
                    obs[team] = _flatten_to_array(obs[team], self._flat_observation_space[team], scope.obs_dtype)
                elif scope.emulate_flat_obs:
                    obs[team] = self.pad_obs
                else:
                    del obs[team]

                if team in rewards:
                    team_infos = {}
                    if team in infos:
                        team_infos = infos[team]

                    # TODO: Handle team infos better
                    rewards[team], infos[team] = self._shape_rewards(
                        rewards[team], dones[team], team_infos, team)

                    # TODO: Improve this check
                    assert type(rewards[team]) in (int, float) \
                        or isinstance(rewards[team], np.number)

                elif scope.emulate_const_num_agents:
                    rewards[team] = 0
                else:
                    del rewards[team]

                if scope.emulate_const_num_agents or team in dones:
                    # TODO: Should dones per team be true on the first tick
                    # or all subsequent ticks as well?
                    dones[team] = self.done or \
                        not any([e in self.agents for e in self._teams[team]])
                else:
                    # TODO: This seems wrong
                    del dones[team]

                # Env might not provide infos key
                #if team in dones:
                if team not in infos:
                    infos[team] = {}

                infos[team]= self._handle_infos(rewards[team], dones[team], infos[team], team)

            # Observation shape test
            if __debug__:
                for team, ob in obs.items():
                    if not self.observation_space(team).contains(ob):
                        raise ValueError(
                            f'observation:\n{ob}\n for agent/team {team} not in '
                            f'observation space:\n{self.observation_space(team)}'
                        )

            return obs, rewards, dones, infos

        @property
        def done(self):
            if len(self.agents) == 0:
                return True
 
            if scope.emulate_const_horizon is None:
                return False
            
            return self._step >= scope.emulate_const_horizon

        @property
        def max_agents(self):
            return len(self._teams)

        @property
        def timers(self):
            return {k: v.serial for k, v in self._timers.items()}

        @property
        def raw_single_observation_space(self):
            '''The observation space of a single agent after featurization but before flattening

            Used by _unpack_batched_obs at the start of the network forward pass
            '''
            return self._raw_observation_space[self.default_team]

        @property
        def single_observation_space(self):
            '''The observation space of a single agent after featurization and flattening'''
            return self._flat_observation_space[self.default_team]

        @property
        def single_action_space(self):
            '''The action space of a single agent after flattening'''
            return self._flat_action_space[self.default_team]

        def close(self):
            if not self.initialized:
                raise exceptions.APIUsageError('close() called before reset()')

            self.env.close()

        @utils.profile
        def observation_space(self, team):
            if team not in self._teams:
                raise exceptions.InvalidAgentError(team, self._teams)

            #Flattened (Box) and cached observation space
            if team in self.obs_space_cache:
                return self.obs_space_cache[team]

            team_obs = self._group_by_team(raw_obs)[team]

            # Initialize obs with dummy postprocessor
            postprocessor = scope.postprocessor_cls(
                self.env, self._teams, team, *scope.postprocessor_args, **scope.postprocessor_kwargs
            )
            postprocessor.reset(team_obs)
            obs = postprocessor.features(team_obs, self._step)

            # Flatten and cache observation space
            # TODO: Rename obs
            self._original_observation_space[team] = raw_local_env.observation_space(team)
            self._raw_observation_space[team] = _make_space_like(obs)
            self._flat_observation_space[team] = _flatten_space(self._raw_observation_space[team])

            # TODO: Add checks on return dtype etc... this should not trigger
            if not self._raw_observation_space[team].contains(obs):
                print({e: self._raw_observation_space[team][0][e].contains(obs[0][e]) for e in obs[0].keys()})
            assert self._raw_observation_space[team].contains(obs)

            # Flatten obs to arrays
            if scope.emulate_flat_obs:
                obs = _flatten_to_array(obs, self._flat_observation_space[team], scope.obs_dtype)

            self.pad_obs = 0 * obs
            obs_space = gym.spaces.Box(
                low=self._obs_min, high=self._obs_max,
                shape=obs.shape, dtype=scope.obs_dtype
            )

            return obs_space

        @utils.profile
        def action_space(self, team):
            if team not in self._teams:
                raise exceptions.InvalidAgentError(team, self._teams)

            #Flattened (MultiDiscrete) and cached action space
            if team in self.atn_space_cache:
                return self.atn_space_cache[team]

            # TODO: Handle variable length teams
            atn_space = gym.spaces.Dict({a: raw_local_env.action_space(a) for a in self._teams[team]})

            self._flat_action_space[team] = _flatten_space(atn_space)
            for agent in self._teams[team]:
                self._agent_action_space[agent] = _flatten_space(atn_space[agent])

            if scope.emulate_flat_atn:
                assert(isinstance(atn_space, gym.Space)), 'Arg must be a gym space'

                if isinstance(atn_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
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

                atn_space = gym.spaces.MultiDiscrete(lens)

            return atn_space

        @utils.profile
        def reset(self, seed=None):
            #Reset the environment and return observations
            obs = self._reset_env(seed)
            obs = self._group_by_team(obs)
            for team_id, team_ob in obs.items():
                self.postprocessors[team_id].reset(team_ob)
                obs[team_id] = self._featurize(team_ob, team_id)

            self.agents = self.env.agents
            self.initialized = True

            # Sort observations according to possible_agents
            # All keys must be present in possible_agents on reset
            obs = {a: obs[a] for a in self.possible_agents}

            orig_obs = obs
            if scope.emulate_flat_obs:
                obs = {team_id: _flatten_to_array(
                    team_ob, self._flat_observation_space[team_id], scope.obs_dtype)
                    for team_id, team_ob in obs.items()}

            if __debug__:
                for agent, ob in obs.items():
                    assert self.observation_space(agent).contains(ob)

                packed_obs = np.stack(list(obs.values()))
                unpacked = unpack_batched_obs(self._flat_observation_space[self.default_team], packed_obs)
                ret = utils._compare_observations(orig_obs, unpacked)
                assert ret, 'Observation packing/unpacking mismatch'

            self._step = 0
            return obs

        @utils.profile
        def step(self, team_actions, **kwargs):
            #Step the environment and return (observations, rewards, dones, infos)
            if not self.initialized:
                raise exceptions.APIUsageError('step() called before reset()')

            if self.done:
                raise exceptions.APIUsageError('step() called after environment is done')

            actions = self._prestep(team_actions)
            obs, rewards, dones, infos = self._step_env(actions)

            # TODO: More env checks here
            if __debug__:
                pass
                #for agent, ob in obs.items():
                #    assert self._original_observation_space[agent].contains(ob), \
                #    f'Agent {agent} observation:\n{ob}\n not in observation'
                #    f'space:\n{self._original_observation_space}'

            self.agents = self.env.agents
            self._step += 1

            # Inject number of agent steps into timer logs
            if not hasattr(self._timers['step'], 'total_agent_steps'):
                self._timers['step'].total_agent_steps = 0
            self._timers['step'].total_agent_steps += len(self.agents)

            obs, rewards, dones, infos = self._poststep(obs, rewards, dones, infos)
            return obs, rewards, dones, infos

    return PufferEnv


class Binding:
    def __init__(self,
            env_cls=None,
            env_creator=None,
            default_args=[],
            default_kwargs={},
            env_name=None,
            postprocessor_cls=Postprocessor,
            postprocessor_args=[],
            postprocessor_kwargs={},
            teams=None,
            emulate_flat_obs=True,
            emulate_flat_atn=True,
            emulate_const_horizon=None,
            emulate_const_num_agents=True,
            suppress_env_prints=__debug__,
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
            postprocessor_cls: Postprocessor subclass to use
            emulate_flat_obs: Whether the observation space requires flattening
            emulate_flat_atn: Whether the action space requires flattening
            emulate_const_horizon: Fixed max horizon for resets, None if not applicable
            emulate_const_num_agents: Whether to pad to len(env.possible_agents) observations
            suppress_env_prints: Whether to consume all environment prints
            obs_dtype: Observation data type
        '''
        if (env_cls is None) == (env_creator is None):
            raise ValueError(
                f'Invalid combination of env_cls={env_cls} and env_creator={env_creator}.'
                ' Specify exactly one one of env_cls (preferred) or env_creator'
            )

        self._env_name = env_name
        self._default_args = default_args
        self._default_kwargs = default_kwargs
        self._obs_dtype = obs_dtype

        self._raw_env_cls = env_cls
        self._raw_env_creator = env_creator
        self._suppress_env_prints = suppress_env_prints

        scope = utils.dotdict(locals())
        del scope['self']

        raw_local_env = self.pz_env_creator()

        raw_obs = _seed_and_reset(raw_local_env, 42)
        self._env_cls = make_puffer_env_cls(scope, raw_local_env, raw_obs)

        local_env = self._env_cls(scope, env=raw_local_env)

        self._default_agent = local_env.possible_agents[0]
        self._max_agents = local_env.max_agents

        self._single_observation_space = local_env.observation_space(self._default_agent)
        self._single_action_space = local_env.action_space(self._default_agent)

        self._featurized_single_observation_space = local_env.single_observation_space
        self._raw_featurized_single_observation_space = local_env.raw_single_observation_space

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

    def pz_env_creator(self):
        '''Returns the partially wrapped PettingZoo env_creator function created by this binding'''
        raw_env = self.raw_env_creator()

        if utils.is_multiagent(raw_env):
            return raw_env

        return GymToPettingZooParallelWrapper(raw_env)

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

    def unpack_batched_obs(self, packed_obs):
        '''Unpack a batch of observations into the original observation space

        Call this funtion in the forward pass of your network
        '''
        return unpack_batched_obs(self._featurized_single_observation_space, packed_obs)

    def pack_obs(self, obs):
        return {k: _flatten_to_array(v, self._featurized_single_observation_space, self._obs_dtype) for k, v in obs.items()}

def unpack_batched_obs(flat_space, packed_obs):
    if not isinstance(flat_space, dict):
        return packed_obs.reshape(packed_obs.shape[0], *flat_space.shape)

    batch = packed_obs.shape[0]

    if () in flat_space:
        return packed_obs.reshape(batch, *flat_space[()].shape)

    batched_obs = {}
    idx = 0

    for key_list, space in flat_space.items():
        current_dict = batched_obs
        inc = int(np.prod(space.shape))

        for key in key_list[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

        last_key = key_list[-1]
        shape = space.shape
        if len(shape) == 0:
            shape = (1,)    

        current_dict[last_key] = packed_obs[:, idx:idx + inc].reshape(batch, *shape)
        idx += inc

    return batched_obs

def _make_space_like(ob):
    if type(ob) == np.ndarray:
        mmin, mmax = utils._get_dtype_bounds(ob.dtype)
        return gym.spaces.Box(
            low=mmin, high=mmax,
            shape=ob.shape, dtype=ob.dtype
        )

    # TODO: Handle Discrete (how to get max?)
    if type(ob) in (tuple, list):
        return gym.spaces.Tuple([_make_space_like(v) for v in ob])

    if type(ob) in (dict, OrderedDict):
        return gym.spaces.Dict({k: _make_space_like(v) for k, v in ob.items()})

    if type(ob) in (int, float):
        # TODO: Tighten bounds
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=())

    raise ValueError(f'Invalid type for featurized obs: {type(ob)}')

def _flatten_space(space):
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

def _flatten_to_array(space_sample, flat_space, dtype=None):
    # TODO: Find a better way to handle Atari
    if type(space_sample) == gym.wrappers.frame_stack.LazyFrames:
       space_sample = np.array(space_sample)

    if () in flat_space:
        if isinstance(space_sample, np.ndarray):
            return space_sample.reshape(*flat_space[()].shape)
        return np.array([space_sample])

    tensors = []
    for key_list in flat_space:
        value = space_sample
        for key in key_list:
            value = value[key]

        if not isinstance(value, np.ndarray):
            value = np.array([value])

        tensors.append(value.ravel())

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

def _seed_and_reset(env, seed):
    try:
        env.seed(seed)
        old_seed=True
    except:
        old_seed=False

    if old_seed:
        obs = env.reset()
    else:
        try:
            obs = env.reset(seed=seed)
        except:
            obs= env.reset()
            warnings.warn('WARNING: Environment does not support seeding.', DeprecationWarning)

    return obs

