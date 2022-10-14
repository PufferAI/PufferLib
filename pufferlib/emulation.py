from pdb import set_trace as T
import numpy as np

from collections.abc import MutableMapping

import gym


def pad_const_num_agents(dummy_ob, pad_to, obs, rewards, dones, infos):
    '''Requires constant integer agent keys'''
    for i in range(pad_to):
        if i not in obs:                                                  
            obs[i]     = dummy_ob                                         
            rewards[i] = 0                                                 
            infos[i]   = {}
            dones[i]   = False

def const_horizon(dones):
    for agent in dones:
        dones[agent] = True

    return dones

def flatten(nested_dict, parent_key=[]):
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + [k]
        if isinstance(v, MutableMapping) or isinstance(v, gym.spaces.Dict):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))

    return {tuple(k): v for k, v in items}

def pack_obs_space(obs_space, dtype=np.float32):
    assert(isinstance(obs_space, gym.Space)), 'Arg must be a gym space'
    flat = flatten(obs_space)

    n = 0
    for e in flat.values():
        n += np.prod(e.shape)

    return gym.spaces.Box(
        low=-2**20, high=2**20,
        shape=(int(n),), dtype=dtype
    )

def pack_atn_space(atn_space):
    assert(isinstance(atn_space, gym.Space)), 'Arg must be a gym space'
    flat = flatten(atn_space)

    lens = []
    for e in flat.values():
        lens.append(e.n)

    return gym.spaces.MultiDiscrete(lens) 

def flatten_ob(ob):
    flat = flatten(ob)
    vals = [e.ravel() for e in flat.values()]
    return np.concatenate(vals)

def pack_obs(obs):
    return {k: flatten_ob(v) for k, v in obs.items()}

def pack_and_batch_obs(obs):
    obs = list(pack_obs(obs).values())
    return np.stack(obs, axis=0)

def unpack_batched_obs(obs_space, packed_obs):
    batch = packed_obs.shape[0]
    obs = {}
    idx = 0

    flat_obs_space = flatten(obs_space)
    for key_list, val in flat_obs_space.items():
        obs_ptr = obs
        for key in key_list:
            if key not in obs_ptr:
                obs_ptr[key] = {}
            if type(obs_ptr[key]) == dict:
                obs_ptr = obs_ptr[key]

        inc = np.prod(val.shape)
        obs_ptr[key] = packed_obs[:, idx:idx + inc].reshape(batch, *val.shape)
        idx = idx + inc

    return obs