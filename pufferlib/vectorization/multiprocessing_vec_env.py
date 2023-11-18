from pdb import set_trace as T
import time

from multiprocessing import Process, Queue
from queue import Empty

from pufferlib import namespace
from pufferlib.vectorization.vec_env import (
    RESET,
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
        num_workers: int = 1,
        envs_per_worker: int = 1,
        batch_size: int = None,
        ) -> None:
    driver_env, multi_env_cls, num_agents = setup(
        env_creator, env_args, env_kwargs, num_workers, envs_per_worker)

    request_queues = [Queue() for _ in range(num_workers)]
    response_queues = [Queue() for _ in range(num_workers)]

    processes = [Process(
        target=_worker_process,
        args=(multi_env_cls, env_creator, env_args, env_kwargs,
              envs_per_worker, request_queues[i], response_queues[i])) 
        for i in range(num_workers)]

    for p in processes:
        p.start()

    return namespace(self,
        processes = processes,
        request_queues = request_queues,
        response_queues = response_queues,
        driver_env = driver_env,
        num_agents = num_agents,
        num_workers = num_workers,
        envs_per_worker = envs_per_worker,
        async_handles = None,
        flag = RESET,
        batch_size = num_workers if batch_size is None else batch_size,
        prev_env_id = [], # Passing explicitly is hard for multiagent and redundant
    )

def _worker_process(multi_env_cls, env_creator, env_args, env_kwargs, n, request_queue, response_queue):
    envs = multi_env_cls(env_creator, env_args, env_kwargs, n=n)

    while True:
        request, args, kwargs = request_queue.get()
        func = getattr(envs, request)
        response = func(*args, **kwargs)
        response_queue.put(response)

def recv(state):
    recv_precheck(state)

    env_id = -1
    recvs = []
    next_env_id = []
    while len(recvs) < state.batch_size:
        env_id = (env_id + 1) % state.num_workers
        queue = state.response_queues[env_id]

        try: # This is ~50% faster than queue.empty() or queue.get_nowait()
            response = queue.get(1e-6)
        except Empty:
            continue

        if response is not None:
            o, r, d, t, i = response
            recvs.append((o, r, d, t, i, env_id))
            next_env_id.append(env_id)

    state.prev_env_id = next_env_id
    return aggregate_recvs(state, recvs)

def send(state, actions):
    send_precheck(state)
    actions = split_actions(state, actions)
    assert len(actions) == state.batch_size
    for i, atns in zip(state.prev_env_id, actions):
        state.request_queues[i].put(("step", [atns], {}))

def async_reset(state, seed=None):
    reset_precheck(state)
    if seed is None:
        for queue in state.request_queues:
            queue.put(("reset", [], {}))
    else:
        for idx, queue in enumerate(state.request_queues):
            queue.put(("reset", [], {"seed": seed+idx}))

def reset(state, seed=None):
    async_reset(state)
    return recv(state)[0]

def step(state, actions):
    send(state, actions)
    return recv(state)

def profile(state):
    for queue in state.request_queues:
        queue.put(("profile", [], {}))

    return aggregate_profiles([queue.get() for queue in state.response_queues])

def put(state, *args, **kwargs):
    for queue in state.request_queues:
        queue.put(("put", args, kwargs))

def get(state, *args, **kwargs):
    '''TODO: fix get'''
    for queue in state.request_queues:
        queue.put(("get", args, kwargs))

    idx = -1
    recvs = []
    while len(recvs) < state.batch_size:
        idx = (idx + 1) % state.num_workers
        queue = state.response_queues[idx]

        if queue.empty():
            continue

        response = queue.get_nowait()
        if response is not None:
            recvs.append(response)

    return recvs


def close(state):
    for queue in state.request_queues:
        queue.put(("close", [], {}))

    for p in state.processes:
        p.terminate()

    for p in state.processes:
        p.join()
