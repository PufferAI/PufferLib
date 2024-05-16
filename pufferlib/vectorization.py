from pdb import set_trace as T

import numpy as np
from math import prod
import gymnasium
import time

from pufferlib import namespace
from pufferlib.environment import PufferEnv
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.exceptions import APIUsageError
import pufferlib.spaces
import pufferlib.exceptions
import pufferlib.emulation

RESET = 0
SEND = 1
RECV = 2

def recv_precheck(state):
    assert state.flag == RECV, 'Call reset before stepping'
    state.flag = SEND

def send_precheck(state):
    assert state.flag == SEND, 'Call reset + recv before send'
    state.flag = RECV

def reset_precheck(state):
    assert state.flag == RESET, 'Call reset only once on initialization'
    state.flag = RECV

def reset(self, seed=None):
    self.async_reset(seed)
    data = self.recv()
    return data[0], data[4]

def step(self, actions):
    actions = np.asarray(actions)
    self.send(actions)
    return self.recv()[:-1]

def vec_prechecks(env_creator, env_args, env_kwargs, num_envs, num_workers, batch_size):
    # TODO: merge into standard vec checks?
    if num_envs < 1:
        raise pufferlib.exceptions.APIUsageError('num_envs must be at least 1')
    # TODO: test case with batch_size = 2, num_envs=6, num_workers=2
    #if num_workers % batch_size != 0:
    #    raise pufferlib.exceptions.APIUsageError('batch_size must be divisible by num_workers')
    if batch_size < 1:
        raise pufferlib.exceptions.APIUsageError('batch_size must be > 0')
    if batch_size > num_envs:
        raise pufferlib.exceptions.APIUsageError('batch_size must be <= num_envs')
    if num_envs % num_workers != 0:
        raise pufferlib.exceptions.APIUsageError('num_envs must be divisible by num_workers')
    if num_envs < num_workers:
        raise pufferlib.exceptions.APIUsageError('num_envs must be >= num_workers')

    if isinstance(env_creator, (list, tuple)) and (env_args or env_kwargs):
        raise pufferlib.exceptions.APIUsageError(
            'env_(kw)args must be empty if env_creator is a list')

class Serial:
    reset = reset
    step = step
 
    def __init__(self, env_creator: callable = None, env_args: list = [], env_kwargs: dict = {},
            num_envs: int = 1, batch_size: int = None, num_workers: int = None):
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs

        vec_prechecks(env_creator, env_args, env_kwargs, num_envs, num_workers, batch_size)
        self.envs = [env_creator(*env_args, **env_kwargs) for _ in range(num_envs)]
        self.driver_env = self.envs[0]
        for env in self.envs:
            assert isinstance(env, (PufferEnv, GymnasiumPufferEnv, PettingZooPufferEnv))
            assert env.single_observation_space == self.driver_env.single_observation_space
            assert env.single_action_space == self.driver_env.single_action_space

        self.single_observation_space = self.driver_env.single_observation_space
        self.single_action_space = self.driver_env.single_action_space
        self.agents_per_env = [env.num_agents for env in self.envs]
        self.agents_per_batch = sum(self.agents_per_env[:batch_size])
        self.num_agents = sum(self.agents_per_env)
        self.agent_ids = np.arange(self.num_agents)
        self.buf = None

    def _assign_buffers(self, buf):
        ptr = 0
        self.buf = buf
        for i, env in enumerate(self.envs):
            end = ptr + self.agents_per_env[i]
            env.buf = namespace(
                observations=buf.observations[ptr:end],
                rewards=buf.rewards[ptr:end],
                terminals=buf.terminals[ptr:end],
                truncations=buf.truncations[ptr:end],
                masks=buf.masks[ptr:end]
            )
            ptr = end

    def async_reset(self, seed=None):
        if self.buf is None:
            self.buf = namespace(
                observations = np.empty(
                    (self.agents_per_batch, *self.single_observation_space.shape),
                    dtype=self.single_observation_space.dtype),
                rewards = np.empty(self.agents_per_batch, dtype=np.float32),
                terminals = np.empty(self.agents_per_batch, dtype=bool),
                truncations = np.empty(self.agents_per_batch, dtype=bool),
                masks = np.empty(self.agents_per_batch, dtype=bool),
            )
            self._assign_buffers(self.buf)

        infos = []
        for idx, env in enumerate(self.envs):
            if seed is None:
                ob, i = env.reset()
            else:
                ob, i = env.reset(seed=hash(1000*seed + idx))
               
            if i:
                infos.append(i)

        buf = self.buf
        buf.rewards[:] = 0
        buf.terminals[:] = False
        buf.truncations[:] = False
        buf.masks[:] = 1
        self.infos = infos

    def send(self, actions):
        rewards, dones, truncateds, self.infos = [], [], [], []
        ptr = 0
        for idx, env in enumerate(self.envs):
            end = ptr + self.agents_per_env[idx]
            atns = actions[ptr:end]
            if env.done:
                o, i = env.reset()
                r = 0
                d = False
                t = False
            else:
                o, r, d, t, i = env.step(atns)

            if i:
                self.infos.append(i)

            ptr = end

    def recv(self):
        buf = self.buf
        return (buf.observations, buf.rewards, buf.terminals, buf.truncations,
            self.infos, self.agent_ids, buf.masks)

    def put(self, *args, **kwargs):
        for e in self.envs:
            e.put(*args, **kwargs)
        
    def get(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.envs]

    def close(self):
        for env in self.envs:
            env.close()

STEP = b"s"
RESET = b"r"
RESET_NONE = b"n"
CLOSE = b"c"
MAIN = b"m"

NEUTRAL = 0
STEP = 1
RESET = 2
RESET_NONE = 3
CLOSE = 4
MAIN = 5
INFO = 6

def _worker_process(env_creator, env_args, env_kwargs, num_envs,
        num_workers, worker_idx, send_pipe, recv_pipe, shm, lock):

    import os


    #import psutil
    #curr_process = psutil.Process()
    #curr_process.cpu_affinity([2*(worker_idx+1)])

    # nice min prioirty
    #import os
    #os.nice(19)

    envs = Serial(env_creator, env_args, env_kwargs, num_envs)
    obs_shape = envs.single_observation_space.shape
    obs_dtype = envs.single_observation_space.dtype
    atn_shape = envs.single_action_space.shape
    atn_dtype = envs.single_action_space.dtype

    # Environments read and write directly to shared memory
    shape = (num_workers, envs.num_agents)
    atn_arr = np.ndarray((*shape, *atn_shape), dtype=atn_dtype, buffer=shm.actions)[worker_idx]
    buf = namespace(
        observations=np.ndarray((*shape, *obs_shape), dtype=obs_dtype, buffer=shm.observations)[worker_idx],
        rewards=np.ndarray(shape, dtype=np.float32, buffer=shm.rewards)[worker_idx],
        terminals=np.ndarray(shape, dtype=bool, buffer=shm.terminals)[worker_idx],
        truncations=np.ndarray(shape, dtype=bool, buffer=shm.truncateds)[worker_idx],
        masks=np.ndarray(shape, dtype=bool, buffer=shm.masks)[worker_idx],
    )
    envs._assign_buffers(buf)

    # TODO: Figure out why not getting called
    envs.reset()

    semaphores=np.ndarray(num_workers, dtype=np.uint8, buffer=shm.semaphores)
    while True:
        #request = recv_pipe.recv_bytes()
        lock.acquire()
        sem = semaphores[worker_idx]
        assert sem != MAIN
        if sem == RESET:
            _, infos = envs.reset()
        elif sem == STEP:
            _, _, _, _, infos, _ = envs.step(atn_arr)
        elif sem == CLOSE:
            print("closing worker", worker_idx)
            #send_pipe.send(None)
            break

        if infos:
            semaphores[worker_idx] = INFO
            #send_pipe.send(infos)
        else:
            semaphores[worker_idx] = MAIN

        #send_pipe.send(envs.infos)

def contiguous_subset(lst, n):
    """
    Checks if the given list contains a contiguous subset of n integers.

    Parameters:
    lst (list of int): The list of integers to check.
    n (int): The length of the contiguous subset to find.

    Returns:
    tuple: A tuple containing the lowest and highest elements of the contiguous subset.
           Returns (None, None) if no such subset exists.
    """
    if len(lst) < n:
        return None

    # Sort the list and remove duplicates
    sorted_lst = sorted(set(lst))

    # Check for contiguous subset of length n
    for i in range(len(sorted_lst) - n + 1):
        if sorted_lst[i + n - 1] - sorted_lst[i] == n - 1:
            return sorted_lst[i]

    return None

class Multiprocessing:
    '''Runs environments in parallel using multiprocessing

    Use this vectorization module for most applications
    '''
    reset = reset
    step = step
    def __init__(self, env_creator: callable = None, env_args: list = [],
            env_kwargs: dict = {}, num_envs: int = 1, num_workers: int = None,
            batch_size: int = None) -> None:
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs

        vec_prechecks(env_creator, env_args, env_kwargs, num_envs, num_workers, batch_size)
        envs_per_worker = num_envs // num_workers
        self.workers_per_batch = batch_size // envs_per_worker
        self.num_workers = num_workers

        # I really didn't want to need a driver process... with mp.shared_memory
        # we can fetch this data from the worker processes and ever perform
        # additional space checks. Unfortunately, SharedMemory has a janky integration
        # with the resource tracker that spams warnings and does not work with
        # forked processes. So for now, RawArray is much more reliable.
        # You can't send a RawArray through a pipe.
        driver_env = env_creator(*env_args, **env_kwargs)
        self.emulated = driver_env.emulated
        self.num_agents = num_agents = driver_env.num_agents * num_envs
        self.agents_per_batch = driver_env.num_agents * batch_size
        agents_per_worker = driver_env.num_agents * envs_per_worker
        obs_space = driver_env.single_observation_space
        obs_shape = obs_space.shape
        obs_dtype = obs_space.dtype
        obs_ctype = np.ctypeslib.as_ctypes_type(obs_dtype)
        atn_space = driver_env.single_action_space
        atn_shape = atn_space.shape
        atn_dtype = atn_space.dtype
        atn_ctype = np.ctypeslib.as_ctypes_type(atn_dtype)

        self.single_observation_space = driver_env.single_observation_space
        self.single_action_space = driver_env.single_action_space
        self.agent_ids = np.arange(num_agents).reshape(num_workers, agents_per_worker)

        # Set process affinity
        #import psutil
        #curr_process = psutil.Process()
        #curr_process.cpu_affinity([0])


        from multiprocessing import RawArray, Semaphore, Lock
        self.shm = namespace(
            observations=RawArray(obs_ctype, num_agents * prod(obs_shape)),
            actions=RawArray(atn_ctype, num_agents * prod(atn_shape)),
            rewards=RawArray('f', num_agents),
            terminals=RawArray('b', num_agents),
            truncateds=RawArray('b', num_agents),
            masks=RawArray('b', num_agents),
            semaphores=RawArray('c', num_workers),
        )
        self.semaphores = [Lock() for _ in range(num_workers)]
        for e in self.semaphores:
            e.acquire()

        shape = (num_workers, agents_per_worker)
        self.obs_batch_shape = (self.agents_per_batch, *obs_shape)
        self.atn_batch_shape = (self.workers_per_batch, agents_per_worker, *atn_shape)
        self.actions = np.ndarray((*shape, *atn_shape), dtype=atn_dtype, buffer=self.shm.actions)
        self.buf = namespace(
            observations=np.ndarray((*shape, *obs_shape), dtype=obs_dtype, buffer=self.shm.observations),
            rewards=np.ndarray(shape, dtype=np.float32, buffer=self.shm.rewards),
            terminals=np.ndarray(shape, dtype=bool, buffer=self.shm.terminals),
            truncations=np.ndarray(shape, dtype=bool, buffer=self.shm.truncateds),
            masks=np.ndarray(shape, dtype=bool, buffer=self.shm.masks),
            semaphores=np.ndarray(num_workers, dtype=np.uint8, buffer=self.shm.semaphores),
        )

        from multiprocessing import Pipe, Process
        self.send_pipes, w_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        w_send_pipes, self.recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        self.recv_pipe_dict = {p: i for i, p in enumerate(self.recv_pipes)}

        self.processes = []
        for i in range(num_workers):
            p = Process(
                target=_worker_process,
                args=(env_creator, env_args, env_kwargs, envs_per_worker,
                    num_workers, i, w_send_pipes[i], w_recv_pipes[i],
                    self.shm, self.semaphores[i])
            )
            p.start()
            self.processes.append(p)

        self.flag = RESET

    @profile
    def recv(self):
        recv_precheck(self)
        start = contiguous_subset(self.ready_workers, self.workers_per_batch)
        while start is None:
            worker = self.waiting_workers.pop(0)
            sem = self.buf.semaphores[worker]
            if sem >= MAIN:
                self.ready_workers.append(worker)
            else:
                self.waiting_workers.append(worker)

            #if sem == INFO:
            #    self.infos[worker] = self.recv_pipes[worker].recv()

            start = contiguous_subset(self.ready_workers, self.workers_per_batch)
            self.start = start

        buf = self.buf
        end = start + self.workers_per_batch
        self.ready_workers = [e for e in self.ready_workers
            if e not in range(start, end)]

        o = buf.observations[start:end].reshape(self.obs_batch_shape)
        r = buf.rewards[start:end].ravel()
        d = buf.terminals[start:end].ravel()
        t = buf.truncations[start:end].ravel()

        #infos = [i for ii in self.infos[start:end] for i in ii]
        #self.infos[start:end] = [[] for _ in range(self.workers_per_batch)]
        infos = []

        agent_ids = self.agent_ids[start:end].ravel()
        m = buf.masks[start:end].ravel()

        return o, r, d, t, infos, agent_ids, m

    @profile
    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.atn_batch_shape)
        
        start = self.start
        end = start + self.workers_per_batch
        self.actions[start:end] = actions
        self.buf.semaphores[start:end] = STEP
        self.waiting_workers.extend(range(start, end))
        for i in range(start, end):
            self.semaphores[i].release()
        #for i in range(start, end):
        #    self.send_pipes[i].send_bytes(STEP)

    def async_reset(self, seed=None):
        self.prev_env_id = []
        reset_precheck(self)
        self.buf.semaphores[:] = RESET
        self.ready_workers = []
        self.waiting_workers = list(range(self.num_workers))
        self.infos = [[] for _ in range(self.num_workers)]

        for i in range(self.num_workers):
            self.semaphores[i].release()

        return
        # TODO: Seed

        if seed is None:
            for pipe in self.send_pipes:
                pipe.send(RESET)
        else:
            for idx, pipe in enumerate(self.send_pipes):
                pipe.send(("reset", [], {"seed": seed+idx}))

    def put(self, *args, **kwargs):
        # TODO: Update this
        for queue in self.request_queues:
            queue.put(("put", args, kwargs))

    def get(self, *args, **kwargs):
        # TODO: Update this
        for queue in self.request_queues:
            queue.put(("get", args, kwargs))

        idx = -1
        recvs = []
        while len(recvs) < self.workers_per_batch // self.envs_per_worker:
            idx = (idx + 1) % self.num_workers
            queue = self.response_queues[idx]

            if queue.empty():
                continue

            response = queue.get()
            if response is not None:
                recvs.append(response)

        return recvs

    def close(self):
        self.buf.semaphores[:] = CLOSE
        for p in self.processes:
            p.terminate()
        '''
        for pipe in self.send_pipes:
            pipe.send_bytes(CLOSE)

        for pipe in self.recv_pipes:
            pipe.recv()
        '''

class Ray():
    '''Runs environments in parallel on multiple processes using Ray

    Use this module for distributed simulation on a cluster. It can also be
    faster than multiprocessing on a single machine for specific environments.
    '''
    reset = reset
    step = step
    #single_observation_space = property(single_observation_space)
    #single_action_space = property(single_action_space)

    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            num_envs: int = 1,
            num_workers: int = None,
            batch_size: int = None,
            ) -> None:
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs

        vec_prechecks(env_creator, env_args, env_kwargs, num_envs, num_workers, batch_size)
        self.env_pool = num_envs != batch_size
        envs_per_worker = num_envs // num_workers
        self.workers_per_batch = batch_size // envs_per_worker
        self.num_workers = num_workers

        driver_env = env_creator(*env_args, **env_kwargs)
        self.emulated = driver_env.emulated
        self.num_agents = num_agents = driver_env.num_agents * num_envs
        self.agents_per_batch = driver_env.num_agents * batch_size
        agents_per_worker = driver_env.num_agents * envs_per_worker
        obs_space = driver_env.single_observation_space
        obs_shape = obs_space.shape
        atn_space = driver_env.single_action_space
        atn_shape = atn_space.shape

        shape = (num_workers, agents_per_worker)
        self.obs_batch_shape = (self.agents_per_batch, *obs_shape)
        self.atn_batch_shape = (self.workers_per_batch, agents_per_worker, *atn_shape)

        self.single_observation_space = driver_env.single_observation_space
        self.single_action_space = driver_env.single_action_space
        self.agent_ids = np.arange(num_agents).reshape(num_workers, agents_per_worker)

        import ray
        if not ray.is_initialized():
            import logging
            ray.init(
                include_dashboard=False,  # WSL Compatibility
                logging_level=logging.ERROR,
            )

        self.envs = [
            ray.remote(Serial).remote(
                env_creator, env_args, env_kwargs, envs_per_worker
            ) for _ in range(num_workers)
        ]

        self.async_handles = None
        self.flag = RESET
        self.ray = ray
        self.prev_env_id = []

    def recv(self):
        recv_precheck(self)
        recvs = []
        next_env_id = []
        workers_per_batch = self.workers_per_batch
        if self.env_pool:
            recvs = self.ray.get(self.async_handles[:workers_per_batch])
            env_id = [_ for _ in range(workers_per_batch)]
        else:
            ready, busy = self.ray.wait(
                self.async_handles, num_returns=workers_per_batch)
            env_id = [self.async_handles.index(e) for e in ready]
            recvs = self.ray.get(ready)

        o, r, d, t, infos, ids, m = zip(*recvs)
        self.prev_env_id = env_id

        infos = [i for ii in infos for i in ii]

        o = np.stack(o, axis=0).reshape(self.obs_batch_shape)
        r = np.stack(r, axis=0).ravel()
        d = np.stack(d, axis=0).ravel()
        t = np.stack(t, axis=0).ravel()
        m = np.stack(m, axis=0).ravel()
        agent_ids = self.agent_ids[env_id].ravel()
        return o, r, d, t, infos, agent_ids, m

    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.atn_batch_shape)
        handles = []
        for i, e in enumerate(self.prev_env_id):
            atns = actions[i]
            env = self.envs[e]
            env.send.remote(atns)
            handles.append(env.recv.remote())

        self.async_handles = handles

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            kwargs = {}
        else:
            kwargs = {"seed": seed}

        handles = []
        for idx, e in enumerate(self.envs):
            e.async_reset.remote(**kwargs)
            handles.append(e.recv.remote())

        self.async_handles = handles

    def put(self, *args, **kwargs):
        for e in self.envs:
            e.put.remote(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.ray.get([e.get.remote(*args, **kwargs) for e in self.envs])

    def close(self):
        self.ray.get([e.close.remote() for e in self.envs])
        self.ray.shutdown()
