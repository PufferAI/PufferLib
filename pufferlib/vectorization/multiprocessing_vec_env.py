from pdb import set_trace as T
import time

import selectors
from multiprocessing import Process, Queue, Manager, Pipe
from queue import Empty

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

    main_send_pipes, work_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
    work_send_pipes, main_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
    
    processes = [Process(
        target=_worker_process,
        args=(multi_env_cls, env_creator, env_args, env_kwargs,
              envs_per_worker, work_send_pipes[i], work_recv_pipes[i]))
        for i in range(num_workers)]

    for p in processes:
        p.start()

    # Register all receive pipes with the selector
    sel = selectors.DefaultSelector()
    for pipe in main_recv_pipes:
        sel.register(pipe, selectors.EVENT_READ)

    return namespace(self,
        processes = processes,
        sel = sel,
        send_pipes = main_send_pipes,
        recv_pipes = main_recv_pipes,
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
        prev_env_id = [],
        synchronous = synchronous,
    )

def _worker_process(multi_env_cls, env_creator, env_args, env_kwargs, n, send_pipe, recv_pipe):
    envs = multi_env_cls(env_creator, env_args, env_kwargs, n=n)

    while True:
        request, args, kwargs = recv_pipe.recv()
        func = getattr(envs, request)
        response = func(*args, **kwargs)
        send_pipe.send(response)

def recv(state):
    recv_precheck(state)

    recvs = []
    next_env_id = []
    if state.synchronous:
        for env_id in range(state.batch_size):
            response_pipe = state.recv_pipes[env_id]
            response = response_pipe.recv()

            o, r, d, t, i = response
            recvs.append((o, r, d, t, i, env_id))
            next_env_id.append(env_id)
    else:
        while len(recvs) < state.workers_per_batch:
            for key, _ in state.sel.select(timeout=None):
                response_pipe = key.fileobj
                env_id = state.recv_pipes.index(response_pipe)

                if response_pipe.poll():  # Check if data is available
                    response = response_pipe.recv()

                    o, r, d, t, i = response
                    recvs.append((o, r, d, t, i, env_id))
                    next_env_id.append(env_id)

                if len(recvs) == state.workers_per_batch:
                    break

    state.prev_env_id = next_env_id
    return aggregate_recvs(state, recvs)

def send(state, actions):
    send_precheck(state)
    actions = split_actions(state, actions)
    assert len(actions) == state.workers_per_batch
    for i, atns in zip(state.prev_env_id, actions):
        state.send_pipes[i].send(("step", [atns], {}))

def async_reset(state, seed=None):
    reset_precheck(state)
    if seed is None:
        for pipe in state.send_pipes:
            pipe.send(("reset", [], {}))
    else:
        for idx, pipe in enumerate(state.send_pipes):
            pipe.send(("reset", [], {"seed": seed+idx}))

def reset(state, seed=None):
    async_reset(state)
    obs, _, _, _, info, env_id, mask = recv(state)
    return obs, info, env_id, mask

def step(state, actions):
    send(state, actions)
    return recv(state)

def profile(state):
    # TODO: Update this
    for queue in state.request_queues:
        queue.put(("profile", [], {}))

    return aggregate_profiles([queue.get() for queue in state.response_queues])

def put(state, *args, **kwargs):
    # TODO: Update this
    for queue in state.request_queues:
        queue.put(("put", args, kwargs))

def get(state, *args, **kwargs):
    # TODO: Update this
    for queue in state.request_queues:
        queue.put(("get", args, kwargs))

    idx = -1
    recvs = []
    while len(recvs) < state.batch_size // state.envs_per_worker:
        idx = (idx + 1) % state.num_workers
        queue = state.response_queues[idx]

        if queue.empty():
            continue

        response = queue.get()
        if response is not None:
            recvs.append(response)

    return recvs


def close(state):
    for pipe in state.send_pipes:
        pipe.send(("close", [], {}))

    for p in state.processes:
        p.terminate()

    for p in state.processes:
        p.join()
