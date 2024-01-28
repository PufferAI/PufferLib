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
    if state.preallocated_obs is None:
        obs_space = state.envs[0].observation_space
        obs_n = obs_space.shape[0]
        n_envs = len(state.envs)

        state.preallocated_obs = np.empty(
            (n_envs, *obs_space.shape), dtype=obs_space.dtype)
        state.preallocated_rewards = np.empty(n_envs, dtype=np.float32)
        state.preallocated_dones = np.empty(n_envs, dtype=bool)
        state.preallocated_truncateds = np.empty(n_envs, dtype=bool)

    infos = []
    for idx, e in enumerate(state.envs):
        if seed is None:
            ob, i = e.reset()
        else:
            ob, i = e.reset(seed=hash(1000*seed + idx))

        i['mask'] = True
        infos.append(i)
        state.preallocated_obs[idx] = ob

    state.preallocated_rewards[:] = 0
    state.preallocated_dones[:] = False
    state.preallocated_truncateds[:] = False

    return (state.preallocated_obs, state.preallocated_rewards,
        state.preallocated_dones, state.preallocated_truncateds, infos)

def step(state, actions):
    infos = []
    for idx, (env, atns) in enumerate(zip(state.envs, actions)):
        if env.done:
            o, i = env.reset()
            state.preallocated_rewards[idx] = 0
            state.preallocated_dones[idx] = False
            state.preallocated_truncateds[idx] = False
        else:
            o, r, d, t, i = env.step(atns)
            state.preallocated_rewards[idx] = r
            state.preallocated_dones[idx] = d
            state.preallocated_truncateds[idx] = t

        i['mask'] = True
        infos.append(i)
        state.preallocated_obs[idx] = o

    return (state.preallocated_obs, state.preallocated_rewards,
        state.preallocated_dones, state.preallocated_truncateds, infos)

class GymMultiEnv:
    __init__ = init
    reset = reset
    step = step
    profile = profile
    put = put
    get = get
    close = close
