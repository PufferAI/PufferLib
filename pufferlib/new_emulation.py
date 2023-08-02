from pdb import set_trace as T

import numpy as np
import warnings

import gym
from collections import OrderedDict

import pufferlib
from pufferlib import utils, exceptions


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

    def infos(self, team_reward, env_done, team_done, team_infos, step):
        if env_done:
            team_infos['return'] = self.epoch_return
            team_infos['length'] = self.epoch_length
            self.done = True
        elif not team_done:
            self.epoch_length += 1
            self.epoch_return += team_reward

        return team_infos



class PufferEnv:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, actions):
        '''Step the environment and return (observations, rewards, dones, infos)'''
        if not self.initialized:
            raise exceptions.APIUsageError('step() called before reset()')

        if self.done:
            raise exceptions.APIUsageError('step() called after environment is done')

        if __debug__:
            for agent, atn in actions.items():
                if agent not in self.agents:
                    raise exceptions.InvalidAgentError(agent, self.agents)

    def render(self):
        pass

    def close(self):
        pass

def pad_agent_data(data, agents, pad_value):
    return {agent: data[agent] if agent in data else pad_value
        for agent in agents}

def check_spaces(data, spaces):
    for k, v in data.items():
        if not spaces(k).contains(v):
            raise ValueError(
                f'Data:\n{v}\n for agent/team {k} not in '
                f'space:\n{spaces(k)}')


def check_teams(env, teams):
    if set(env.possible_agents) != {item for team in teams.values() for item in team}:
        raise ValueError(f'Invalid teams: {teams} for possible_agents: {env.possible_agents}')

def group_into_teams(teams, *args):
    grouped_data = []

    for agent_data in args:
        if __debug__ and set(agent_data) != {item for team in teams.values() for item in team}:
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

def unpack_actions(actions, agent_ids, flat_space):
    split_actions = {}
    for team_id, team in agent_ids.items():
        # TODO: Assert all keys present since actions are padded
        team_atns = np.split(actions[team_id], len(team))
        for agent_id, atns in zip(team, team_atns):
            split_actions[agent_id] = atns

    actions = split_actions
    for k in list(actions):
        if k not in agent_ids:
            del(actions[k])
            continue

        flat = actions[k]

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

def convert_to_multidiscrete(flat_space):
    lens = []
    for e in flat_space.values():
        if isinstance(e, gym.spaces.Discrete):
            lens.append(e.n)
        elif isinstance(e, gym.spaces.MultiDiscrete):
            lens += e.nvec.tolist()
        else:
            raise ValueError(f'Invalid action space: {e}')

    return gym.spaces.MultiDiscrete(lens)

def make_space_like(ob):
    if type(ob) == np.ndarray:
        mmin, mmax = utils._get_dtype_bounds(ob.dtype)
        return gym.spaces.Box(
            low=mmin, high=mmax,
            shape=ob.shape, dtype=ob.dtype
        )

    # TODO: Handle Discrete (how to get max?)
    if type(ob) in (tuple, list):
        return gym.spaces.Tuple([make_space_like(v) for v in ob])

    if type(ob) in (dict, OrderedDict):
        return gym.spaces.Dict({k: make_space_like(v) for k, v in ob.items()})

    if type(ob) in (int, float):
        # TODO: Tighten bounds
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=())

    raise ValueError(f'Invalid type for featurized obs: {type(ob)}')

def flatten_space(space):
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

def flatten_to_array(space_sample, flat_space, dtype=None):
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
            try:
                value = value[key]
            except:
                T()

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