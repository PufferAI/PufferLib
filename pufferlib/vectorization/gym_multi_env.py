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
    for idx, e in enumerate(state.envs):
        if seed is None:
            ob = e.reset()
        else:
            ob = e.reset(seed=hash(1000*seed + idx))

        if state.preallocated_obs is None:
            state.preallocated_obs = np.empty(
                (len(state.envs), *ob.shape), dtype=ob.dtype)

        state.preallocated_obs[idx] = ob

    rewards = [0] * len(state.preallocated_obs)
    dones = [False] * len(state.preallocated_obs)
    infos = [{} for _ in state.envs]

    return state.preallocated_obs, rewards, dones, infos

def step(state, actions):
    rewards, dones, infos = [], [], []

    for idx, (env, atns) in enumerate(zip(state.envs, actions)):
        if env.done:
            o  = env.reset()
            rewards.append(0)
            dones.append(False)
            infos.append({})
        else:
            o, r, d, i = env.step(atns)
            rewards.append(r)
            dones.append(d)
            infos.append(i)

        state.preallocated_obs[idx] = o

    return state.preallocated_obs, rewards, dones, infos

class GymMultiEnv:
    __init__ = init
    reset = reset
    step = step
    profile = profile
    put = put
    get = get
    close = close
