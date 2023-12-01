from pdb import set_trace as T

import gym

from pufferlib import namespace
from pufferlib.vectorization.vec_env import (
    RESET,
    calc_scale_params,
    setup,
    single_observation_space,
    single_action_space,
    single_action_space,
    structured_observation_space,
    flat_observation_space,
    unpack_batched_obs,
    reset_precheck,
    recv_precheck,
    send_precheck,
    aggregate_recvs,
    split_actions,
    aggregate_profiles,
)

def init(self: object = None,
        env_creator: callable = None,
        env_args: list = [],
        env_kwargs: dict = {},
        num_envs: int = 1,
        envs_per_worker: int = 1,
        envs_per_batch: int = None,
        synchronous: bool = False,
        ) -> None:
    driver_env, multi_env_cls, agents_per_env = setup(
        env_creator, env_args, env_kwargs)
    num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker = calc_scale_params(
        num_envs, envs_per_batch, envs_per_worker, agents_per_env)

    multi_envs = [
        multi_env_cls(
            env_creator, env_args, env_kwargs, envs_per_worker,
        ) for _ in range(num_workers)
    ]

    return namespace(self,
        multi_envs = multi_envs,
        driver_env = driver_env,
        num_envs = num_envs,
        num_workers = num_workers,
        workers_per_batch = workers_per_batch,
        envs_per_batch = envs_per_batch,
        envs_per_worker = envs_per_worker,
        agents_per_batch = agents_per_batch,
        agents_per_worker = agents_per_worker,
        agents_per_env = agents_per_env,
        async_handles = None,
        flag = RESET,
    )

def recv(state):
    recv_precheck(state)
    recvs = [(o, r, d, t, i, env_id) for (o, r, d, t, i), env_id
        in zip(state.data, range(state.workers_per_batch))]
    return aggregate_recvs(state, recvs)

def send(state, actions):
    send_precheck(state)
    actions = split_actions(state, actions)
    state.data = [e.step(a) for e, a in zip(state.multi_envs, actions)]

def async_reset(state, seed=None):
    reset_precheck(state)
    if seed is None:
        state.data = [e.reset() for e in state.multi_envs]
    else:
        state.data = [e.reset(seed=seed+idx) for idx, e in enumerate(state.multi_envs)]

def reset(state, seed=None):
    async_reset(state)
    obs, _, _, _, info, env_id, mask = recv(state)
    return obs, info, env_id, mask

def step(state, actions):
    send(state, actions)
    return recv(state)

def profile(state):
    return aggregate_profiles([e.profile() for e in state.multi_envs])

def put(state, *args, **kwargs):
    for e in state.multi_envs:
        e.put(*args, **kwargs)

def get(state, *args, **kwargs):
    return [e.get(*args, **kwargs) for e in state.multi_envs]

def close(state):
    for e in state.multi_envs:
        e.close()
