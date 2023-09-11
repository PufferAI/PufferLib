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

    ptr = 0
    for idx, e in enumerate(state.envs):
        if seed is None:
            obs = e.reset()
        else:
            obs = e.reset(seed=hash(1000*seed + idx))

        state.agent_keys.append(list(obs.keys()))

        if state.preallocated_obs is None:
            ob = obs[list(obs.keys())[0]]
            state.preallocated_obs = np.empty((len(state.envs)*len(obs), *ob.shape), dtype=ob.dtype)

        for o in obs.values():
            state.preallocated_obs[ptr] = o
            ptr += 1

    rewards = [0] * len(state.preallocated_obs)
    dones = [False] * len(state.preallocated_obs)
    infos = [
        {agent_id: {} for agent_id in state.envs[0].possible_agents}
        for _ in state.envs
    ]

    return state.preallocated_obs, rewards, dones, infos

def step(state, actions):
    actions = np.array_split(actions, len(state.envs))
    rewards, dones, infos = [], [], []

    ptr = 0
    for idx, (a_keys, env, atns) in enumerate(zip(state.agent_keys, state.envs, actions)):
        if env.done:
            o  = env.reset()
            num_agents = len(env.possible_agents)
            rewards.extend([0] * num_agents)
            dones.extend([False] * num_agents)
            infos.append({agent_id: {} for agent_id in env.possible_agents})
        else:
            assert len(a_keys) == len(atns)
            atns = dict(zip(a_keys, atns))
            o, r, d, i= env.step(atns)
            rewards.extend(r.values())
            dones.extend(d.values())
            infos.append(i)

        state.agent_keys[idx] = list(o.keys())

        for oo in o.values():
            state.preallocated_obs[ptr] = oo
            ptr += 1

    return state.preallocated_obs, rewards, dones, infos

class PettingZooMultiEnv:
    __init__ = init
    reset = reset
    step = step
    profile = profile
    put = put
    get = get
    close = close
