from pdb import set_trace as T

import numpy as np
import warnings

import gym
import gymnasium
import inspect
from collections import OrderedDict
from collections.abc import Iterable

import pufferlib
import pufferlib.spaces
from pufferlib import utils, exceptions
from pufferlib.extensions import flatten, unflatten

DICT = 0
LIST = 1
TUPLE = 2
VALUE = 3


class Postprocessor:
    '''Modify data before it is returned from or passed to the environment

    For multi-agent environments, each agent has its own stateful postprocessor.
    '''
    def __init__(self, env, is_multiagent, agent_id=None):
        '''Postprocessors provide full access to the environment
        
        This means you can use them to cheat. Don't blame us if you do.
        '''
        self.env = env
        self.agent_id = agent_id
        self.is_multiagent = is_multiagent

    @property
    def observation_space(self):
        '''The space of observations output by the postprocessor
        
        You will have to implement this function if Postprocessor.observation
        modifies the structure of observations. Defaults to the env's obs space.

        PufferLib supports heterogeneous observation spaces for multi-agent environments,
        provided that your postprocessor pads or otherwise cannonicalizes the observations.
        '''
        if self.is_multiagent:
            return self.env.observation_space(self.env.possible_agents[0])
        return self.env.observation_space

    def reset(self, observation):
        '''Called at the beginning of each episode'''
        return

    def observation(self, observation):
        '''Called on each observation after it is returned by the environment
        
        You must override Postprocessor.observation_space if this function
        changes the structure of observations.
        '''
        return observation

    def action(self, action):
        '''Called on each action before it is passed to the environment
        
        Actions output by your policy do not need to match the action space,
        but they must be compatible after this function is called.
        '''
        return action

    def reward_done_truncated_info(self, reward, done, truncated, info):
        '''Called on the reward, done, truncated, and info after they are returned by the environment'''
        return reward, done, truncated, info


class BasicPostprocessor(Postprocessor):
    '''Basic postprocessor that injects returns and lengths information into infos and
    provides an option to pad to a maximum episode length. Works for single-agent and
    team-based multi-agent environments'''
    def reset(self, obs):
        self.epoch_return = 0
        self.epoch_length = 0
        self.done = False

    def reward_done_truncated_info(self, reward, done, truncated, info):
        if isinstance(reward, (list, np.ndarray)):
            reward = sum(reward.values())

        # Env is done
        if self.done:
            return reward, done, truncated, info

        self.epoch_length += 1
        self.epoch_return += reward

        if done or truncated:
            info['return'] = self.epoch_return
            info['length'] = self.epoch_length
            self.done = True

        return reward, done, truncated, info

class GymnasiumPufferEnv(gym.Env):
    def __init__(self, env=None, env_creator=None, env_args=[], env_kwargs={},
            postprocessor_cls=Postprocessor):
        self.env = make_object(env, env_creator, env_args, env_kwargs)
        self.postprocessor = postprocessor_cls(self.env, is_multiagent=False)

        self.initialized = False
        self.done = True

        self.is_observation_checked = False
        self.is_action_checked = False

        # Cache the observation and action spaces
        self.observation_space
        self.action_space

    @property
    def observation_space(self):
        '''Returns a flattened, single-tensor observation space'''
        self.structured_observation_space = self.postprocessor.observation_space

        # Flatten the featurized observation space and store
        # it for use in step. Return a box space for the user
        self.flat_observation_space, self.flat_observation_structure, self.box_observation_space, self.pad_observation = (
            make_flat_and_box_obs_space(self.structured_observation_space))

        return self.box_observation_space

    @property
    def action_space(self):
        '''Returns a flattened, multi-discrete action space'''
        self.structured_action_space = self.env.action_space
        self.flat_action_structure = flatten_structure(self.structured_action_space.sample())

        # Store a flat version of the action space for use in step. Return a multidiscrete version for the user
        self.flat_action_space, self.multidiscrete_action_space = (
            make_flat_and_multidiscrete_atn_space(self.env.action_space))

        return self.multidiscrete_action_space

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, seed=None):
        self.initialized = True
        self.done = False

        ob, info = _seed_and_reset(self.env, seed)

        # Call user featurizer and flatten the observations
        self.postprocessor.reset(ob)
        processed_ob = concatenate(flatten(self.postprocessor.observation(ob)))

        if __debug__:
            if not self.is_observation_checked:
                self.is_observation_checked = check_space(
                    processed_ob, self.box_observation_space)

        return processed_ob, info
 
    def step(self, action):
        '''Execute an action and return (observation, reward, done, info)'''
        if not self.initialized:
            raise exceptions.APIUsageError('step() called before reset()')
        if self.done:
            raise exceptions.APIUsageError('step() called after environment is done')
 
        action = self.postprocessor.action(action)

        if __debug__:
            if not self.is_action_checked:
                self.is_action_checked = check_space(
                    action, self.multidiscrete_action_space)

        # Unpack actions from multidiscrete into the original action space
        action = unflatten(
            split(
                action, self.flat_action_space, batched=False
            ), self.flat_action_structure
        )

        ob, reward, done, truncated, info = self.env.step(action)

        # Call user postprocessors and flatten the observations
        ob = concatenate(flatten(self.postprocessor.observation(ob)))
        reward, done, truncated, info = self.postprocessor.reward_done_truncated_info(reward, done, truncated, info)
                   
        self.done = done
        return ob, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def unpack_batched_obs(self, batched_obs):
        return unpack_batched_obs(batched_obs, self.flat_observation_space, self.flat_observation_structure)


class PettingZooPufferEnv:
    def __init__(self, env=None, env_creator=None, env_args=[], env_kwargs={},
                 postprocessor_cls=Postprocessor, postprocessor_kwargs={}, teams=None):
        self.env = make_object(env, env_creator, env_args, env_kwargs)
        self.initialized = False
        self.all_done = True

        self.is_observation_checked = False
        self.is_action_checked = False

        self.possible_agents = self.env.possible_agents if teams is None else list(teams.keys())
        self.teams = teams

        self.postprocessors = {agent: postprocessor_cls(
                self.env, is_multiagent=True, agent_id=agent, **postprocessor_kwargs)
            for agent in self.possible_agents}

        # Cache the observation and action spaces
        self.observation_space(self.possible_agents[0])
        self.action_space(self.possible_agents[0])

    @property
    def agents(self):
        return self.env.agents

    @property
    def done(self):
        return len(self.agents) == 0 or self.all_done

    @property
    def single_observation_space(self):
        return self.box_observation_space

    @property
    def single_action_space(self):
        return self.multidiscrete_action_space

    def observation_space(self, agent):
        '''Returns the observation space for a single agent'''
        if agent not in self.possible_agents:
            raise pufferlib.exceptions.InvalidAgentError(agent, self.possible_agents)

        # Make a gym space defining observations for the whole team
        if self.teams is not None:
            obs_space = make_team_space(
                self.env.observation_space, self.teams[agent])
        else:
            obs_space = self.env.observation_space(agent)

        # Call user featurizer and create a corresponding gym space
        self.structured_observation_space = self.postprocessors[agent].observation_space

        # Flatten the featurized observation space and store it for use in step. Return a box space for the user
        self.flat_observation_space, self.flat_observation_structure, self.box_observation_space, self.pad_observation = (
            make_flat_and_box_obs_space(self.structured_observation_space))

        return self.box_observation_space 

    def action_space(self, agent):
        '''Returns the action space for a single agent'''
        if agent not in self.possible_agents:
            raise pufferlib.exceptions.InvalidAgentError(agent, self.possible_agents)

        # Make a gym space defining actions for the whole team
        if self.teams is not None:
            atn_space = make_team_space(
                self.env.action_space, self.teams[agent])
        else:
            atn_space = self.env.action_space(agent)

        self.structured_action_space = atn_space
        self.flat_action_structure = flatten_structure(atn_space.sample())

        # Store a flat version of the action space for use in step. Return a multidiscrete version for the user
        self.flat_action_space, self.multidiscrete_action_space = make_flat_and_multidiscrete_atn_space(atn_space)

        return self.multidiscrete_action_space

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.initialized = True
        self.all_done = False

        # Group observations into teams
        if self.teams is not None:
            obs = group_into_teams(self.teams, obs)

        # Call user featurizer and flatten the observations
        postprocessed_obs = {}
        ob = list(obs.values())[0]
        for agent in self.possible_agents:
            post = self.postprocessors[agent]
            post.reset(ob)
            if agent in obs:
                ob = obs[agent]
                postprocessed_obs[agent] = concatenate(flatten(post.observation(ob)))

        if __debug__:
            if not self.is_observation_checked:
                self.is_observation_checked = check_space(
                    next(iter(postprocessed_obs.values())),
                    self.box_observation_space
                )

        padded_obs = pad_agent_data(postprocessed_obs,
            self.possible_agents, self.pad_observation)

        # Mask out missing agents
        padded_infos = {}
        for agent in self.possible_agents:
            if agent not in info:
                padded_infos[agent] = {}
            else:
                padded_infos[agent] = info[agent]
            padded_infos[agent]['mask'] = agent in obs

        return padded_obs, padded_infos

    def step(self, actions):
        '''Step the environment and return (observations, rewards, dones, infos)'''
        if not self.initialized:
            raise exceptions.APIUsageError('step() called before reset()')
        if self.done:
            raise exceptions.APIUsageError('step() called after environment is done')

        # Postprocess actions and validate action spaces
        for agent in actions:
            if __debug__:
                if agent not in self.possible_agents:
                    raise exceptions.InvalidAgentError(agent, self.agents)

            actions[agent] = self.postprocessors[agent].action(actions[agent])

        if __debug__:
            if not self.is_action_checked:
                self.is_action_checked = check_space(
                    next(iter(actions.values())),
                    self.multidiscrete_action_space
                )

        # Unpack actions from multidiscrete into the original action space
        unpacked_actions = {}
        for agent, atn in actions.items():
            if agent in self.agents:
                unpacked_actions[agent] = unflatten(
                    split(atn, self.flat_action_space, batched=False),
                    self.flat_action_structure
                )

        if self.teams is not None:
            unpacked_actions = ungroup_from_teams(self.teams, unpacked_actions)

        obs, rewards, dones, truncateds, infos = self.env.step(unpacked_actions)
        # TODO: Can add this assert once NMMO Horizon is ported to puffer
        # assert all(dones.values()) == (len(self.env.agents) == 0)

        if self.teams is not None:
            obs, rewards, truncateds, dones = group_into_teams(self.teams, obs, rewards, truncateds, dones)

        # Call user postprocessors and flatten the observations
        for agent in obs:
            obs[agent] = concatenate(flatten(self.postprocessors[agent].observation(obs[agent])))
            rewards[agent], dones[agent], truncateds[agent], infos[agent] = self.postprocessors[agent].reward_done_truncated_info(
                rewards[agent], dones[agent], truncateds[agent], infos[agent])
     
        self.all_done = all(dones.values())

        # Mask out missing agents
        for agent in self.possible_agents:
            if agent not in infos:
                infos[agent] = {}
            else:
                infos[agent] = infos[agent]
            infos[agent]['mask'] = agent in obs

        obs, rewards, dones, truncateds = pad_to_const_num_agents(
            self.env.possible_agents, obs, rewards, dones, truncateds, self.pad_observation)

        return obs, rewards, dones, truncateds, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def unpack_batched_obs(self, batched_obs):
        return unpack_batched_obs(batched_obs, self.flat_observation_space, self.flat_observation_structure)


def unpack_batched_obs(batched_obs, flat_observation_space, flat_observation_structure):
    unpacked = split(batched_obs, flat_observation_space, batched=True)
    unflattened = unflatten(unpacked, flat_observation_structure)
    return unflattened

def make_object(object_instance=None, object_creator=None, creator_args=[], creator_kwargs={}):
    if (object_instance is None) == (object_creator is None):
        raise ValueError('Exactly one of object_instance or object_creator must be provided')

    if object_instance is not None:
        if callable(object_instance) or inspect.isclass(object_instance):
            raise TypeError('object_instance must be an instance, not a function or class')
        return object_instance

    if object_creator is not None:
        if not callable(object_creator):
            raise TypeError('object_creator must be a callable')
        
        if creator_args is None:
            creator_args = []

        if creator_kwargs is None:
            creator_kwargs = {}

        return object_creator(*creator_args, **creator_kwargs)

def pad_agent_data(data, agents, pad_value):
    return {agent: data[agent] if agent in data else pad_value
        for agent in agents}
    
def pad_to_const_num_agents(agents, obs, rewards, dones, truncateds, pad_obs):
    padded_obs = pad_agent_data(obs, agents, pad_obs)
    rewards = pad_agent_data(rewards, agents, 0)
    dones = pad_agent_data(dones, agents, False)
    truncateds = pad_agent_data(truncateds, agents, False)
    return padded_obs, rewards, dones, truncateds

def make_flat_and_multidiscrete_atn_space(atn_space):
    flat_action_space = flatten_space(atn_space)
    if len(flat_action_space) == 1:
        return flat_action_space, list(flat_action_space.values())[0]
    multidiscrete_space = convert_to_multidiscrete(flat_action_space)
    return flat_action_space, multidiscrete_space


def make_flat_and_box_obs_space(obs_space):
    obs = obs_space.sample()
    flat_observation_structure = flatten_structure(obs)

    flat_observation_space = flatten_space(obs_space)  
    obs = obs_space.sample()

    flat_observation = concatenate(flatten(obs))

    mmin, mmax = pufferlib.utils._get_dtype_bounds(flat_observation.dtype)
    pad_obs = flat_observation * 0
    box_obs_space = gymnasium.spaces.Box(
        low=mmin, high=mmax,
        shape=flat_observation.shape, dtype=flat_observation.dtype
    )

    return flat_observation_space, flat_observation_structure, box_obs_space, pad_obs


def make_featurized_obs_and_space(obs_space, postprocessor):
    obs_sample = obs_space.sample()
    featurized_obs = postprocessor.observation(obs_sample)
    featurized_obs_space = make_space_like(featurized_obs)
    return featurized_obs_space, featurized_obs

def make_team_space(observation_space, agents):
    return gymnasium.spaces.Dict({agent: observation_space(agent) for agent in agents})

def check_space(data, space):
    try:
        contains = space.contains(data)
    except:
        raise ValueError(
            f'Error checking space {space} with sample :\n{data}')

    if not contains:
        raise ValueError(
            f'Data:\n{data}\n not in space:\n{space}')
    
    return True

def check_teams(env, teams):
    if set(env.possible_agents) != {item for team in teams.values() for item in team}:
        raise ValueError(f'Invalid teams: {teams} for possible_agents: {env.possible_agents}')

def group_into_teams(teams, *args):
    grouped_data = []

    for agent_data in args:
        if __debug__:
            if set(agent_data) != {item for team in teams.values() for item in team}:
                raise ValueError(f'Invalid teams: {teams} for agents: {set(agent_data)}')

        team_data = {}
        for team_id, team in teams.items():
            team_data[team_id] = {}
            for agent_id in team:
                if agent_id in agent_data:
                    team_data[team_id][agent_id] = agent_data[agent_id]

        grouped_data.append(team_data)

    if len(grouped_data) == 1:
        return grouped_data[0]

    return grouped_data

def ungroup_from_teams(team_data):
    agent_data = {}
    for team in team_data.values():
        for agent_id, data in team.items():
            agent_data[agent_id] = data
    return agent_data


def flatten_structure(data):
    structure = []
    
    def helper(d):
        if isinstance(d, dict):
            structure.append(DICT)
            structure.append(len(d))
            for key, value in d.items():
                structure.append(key)
                helper(value)
        elif isinstance(d, list):
            structure.append(LIST)
            structure.append(len(d))
            for item in d:
                helper(item)
        elif isinstance(d, tuple):
            structure.append(TUPLE)
            structure.append(len(d))
            for item in d:
                helper(item)
        else:
            structure.append(VALUE)
    
    helper(data)
    return structure

def flatten_space(space):
    def _recursion_helper(current, key):
        if isinstance(current, pufferlib.spaces.Tuple):
            for idx, elem in enumerate(current):
                _recursion_helper(elem, f'{key}T{idx}.')
        elif isinstance(current, pufferlib.spaces.Dict):
            for k, value in current.items():
                _recursion_helper(value, f'{key}D{k}.')
        else:
            flat[f'{key}V'] = current

    flat = {}
    _recursion_helper(space, '')
    return flat

def concatenate(flat_sample):
    if len(flat_sample) == 1:
        flat_sample = flat_sample[0]
        if isinstance(flat_sample,(np.ndarray,
                gymnasium.wrappers.frame_stack.LazyFrames)):
            return flat_sample
        return np.array([flat_sample])
    return np.concatenate([
        e.ravel() if isinstance(e, np.ndarray) else np.array([e])
        for e in flat_sample]
    )

def split(stacked_sample, flat_space, batched=True):
    if not isinstance(stacked_sample, Iterable):
        return [stacked_sample]

    if batched:
        batch = stacked_sample.shape[0]

    leaves = []
    ptr = 0
    for subspace in flat_space.values():
        shape = subspace.shape
        typ = subspace.dtype
        sz = int(np.prod(shape))

        if shape == ():
            shape = (1,)

        if batched:
            samp = stacked_sample[:, ptr:ptr+sz].reshape(batch, *shape)
        else:
            samp = stacked_sample[ptr:ptr+sz].reshape(*shape).astype(typ)
            if isinstance(subspace, pufferlib.spaces.Discrete):
                samp = int(samp[0])

        leaves.append(samp)
        ptr += sz

    return leaves

def convert_to_multidiscrete(flat_space):
    lens = []
    for e in flat_space.values():
        if isinstance(e, pufferlib.spaces.Discrete):
            lens.append(e.n)
        elif isinstance(e, pufferlib.spaces.MultiDiscrete):
            lens += e.nvec.tolist()
        else:
            raise ValueError(f'Invalid action space: {e}')

    return gymnasium.spaces.MultiDiscrete(lens)

def make_space_like(ob):
    if type(ob) == np.ndarray:
        mmin, mmax = utils._get_dtype_bounds(ob.dtype)
        return gymnasium.spaces.Box(
            low=mmin, high=mmax,
            shape=ob.shape, dtype=ob.dtype
        )

    # TODO: Handle Discrete (how to get max?)
    if type(ob) in (tuple, list):
        return gymnasium.spaces.Tuple([make_space_like(v) for v in ob])

    if type(ob) in (dict, OrderedDict):
        return gymnasium.spaces.Dict({k: make_space_like(v) for k, v in ob.items()})

    if type(ob) in (int, float):
        # TODO: Tighten bounds
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=())

    raise ValueError(f'Invalid type for featurized obs: {type(ob)}')
    
def _seed_and_reset(env, seed):
    if seed is None:
        # Gym bug: does not reset env correctly
        # when seed is passed as explicit None
        return env.reset()

    try:
        obs, info = env.reset(seed=seed)
    except:
        try:
            env.seed(seed)
            obs, info = env.reset()
        except:
            obs, info = env.reset()
            warnings.warn('WARNING: Environment does not support seeding.', DeprecationWarning)

    return obs, info
