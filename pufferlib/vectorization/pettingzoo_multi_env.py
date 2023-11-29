from pdb import set_trace as T

import numpy as np

from pufferlib.vectorization.multi_env import (
    init,
    profile,
    put,
    get,
    close,
)


def reset(state, seed=None):
    state.agent_keys = []
    infos = []

    ptr = 0
    for idx, e in enumerate(state.envs):
        if seed is None:
            obs, i = e.reset()
        else:
            obs, i = e.reset(seed=hash(1000*seed + idx))

        state.agent_keys.append(list(obs.keys()))
        infos.append(i)

        if state.preallocated_obs is None:
            ob = obs[list(obs.keys())[0]]
            state.preallocated_obs = np.empty((len(state.envs)*len(obs), *ob.shape), dtype=ob.dtype)

        for o in obs.values():
            state.preallocated_obs[ptr] = o
            ptr += 1

    rewards = [0] * len(state.preallocated_obs)
    dones = [False] * len(state.preallocated_obs)
    truncateds = [False] * len(state.preallocated_obs)
    return state.preallocated_obs, rewards, dones, truncateds, infos

def step(state, actions):
    actions = np.array_split(actions, len(state.envs))
    rewards, dones, truncateds, infos = [], [], [], []

    ptr = 0
    for idx, (a_keys, env, atns) in enumerate(zip(state.agent_keys, state.envs, actions)):
        if env.done:
            o, i = env.reset()
            num_agents = len(env.possible_agents)
            rewards.extend([0] * num_agents)
            dones.extend([False] * num_agents)
            truncateds.extend([False] * num_agents)
        else:
            assert len(a_keys) == len(atns)
            atns = dict(zip(a_keys, atns))
            o, r, d, t, i = env.step(atns)
            rewards.extend(r.values())
            dones.extend(d.values())
            truncateds.extend(t.values())

        infos.append(i)
        state.agent_keys[idx] = list(o.keys())

        for oo in o.values():
            state.preallocated_obs[ptr] = oo
            ptr += 1

    return state.preallocated_obs, rewards, dones, truncateds, infos

class PettingZooMultiEnv:
    __init__ = init
    reset = reset
    step = step
    profile = profile
    put = put
    get = get
    close = close
