from pdb import set_trace as T

import numpy as np
import time

from pufferlib import namespace
from pufferlib.environment import PufferEnv
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.exceptions import APIUsageError

NEUTRAL = 0
RESET = 0
STEP = 1
SEND = 2
RECV = 3
CLOSE = 4
MAIN = 5
INFO = 6

def recv_precheck(vecenv):
    if vecenv.flag != RECV:
        raise APIUsageError('Call reset before stepping')

    vecenv.flag = SEND

def send_precheck(vecenv):
    if vecenv.flag != SEND:
        raise APIUsageError('Call (async) reset + recv before sending')

    vecenv.flag = RECV

def reset(vecenv, seed=42):
    vecenv.async_reset(seed)
    obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
    return obs, infos

def step(vecenv, actions):
    actions = np.asarray(actions)
    vecenv.send(actions)
    obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
    return obs, rewards, terminals, truncations, infos, env_ids
 
class Serial:
    reset = reset
    step = step
 
    def __init__(self, env_creators, env_args, env_kwargs, num_envs, **kwargs):
        self.envs = [creator(*args, **kwargs) for (creator, args, kwargs)
            in zip(env_creators, env_args, env_kwargs)]

        self.driver_env = driver = self.envs[0]
        check_envs(self.envs, self.driver_env)
        self.single_observation_space = driver.single_observation_space
        self.single_action_space = driver.single_action_space
        self.agents_per_env = [env.num_agents for env in self.envs]
        self.agents_per_batch = sum(self.agents_per_env)
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

    def async_reset(self, seed=42):
        seed = make_seeds(seed, len(self.envs))

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
        for env, s in zip(self.envs, seed):
            ob, i = env.reset(seed=s)
               
            if i:
                infos.append(i)

        buf = self.buf
        buf.rewards[:] = 0
        buf.terminals[:] = False
        buf.truncations[:] = False
        buf.masks[:] = 1
        self.infos = infos

    def send(self, actions):
        actions = np.asarray(actions)
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

    def close(self):
        for env in self.envs:
            env.close()

def _worker_process(env_creators, env_args, env_kwargs, num_envs,
        num_workers, worker_idx, send_pipe, recv_pipe, shm):
    envs = Serial(env_creators, env_args, env_kwargs, num_envs)
    obs_shape = envs.single_observation_space.shape
    obs_dtype = envs.single_observation_space.dtype
    atn_shape = envs.single_action_space.shape
    atn_dtype = envs.single_action_space.dtype

    # Environments read and write directly to shared memory
    shape = (num_workers, envs.num_agents)
    atn_arr = np.ndarray((*shape, *atn_shape),
        dtype=atn_dtype, buffer=shm.actions)[worker_idx]
    buf = namespace(
        observations=np.ndarray((*shape, *obs_shape),
            dtype=obs_dtype, buffer=shm.observations)[worker_idx],
        rewards=np.ndarray(shape, dtype=np.float32, buffer=shm.rewards)[worker_idx],
        terminals=np.ndarray(shape, dtype=bool, buffer=shm.terminals)[worker_idx],
        truncations=np.ndarray(shape, dtype=bool, buffer=shm.truncateds)[worker_idx],
        masks=np.ndarray(shape, dtype=bool, buffer=shm.masks)[worker_idx],
    )
    envs._assign_buffers(buf)

    semaphores=np.ndarray(num_workers, dtype=np.uint8, buffer=shm.semaphores)
    start = time.time()
    while True:
        sem = semaphores[worker_idx]
        if sem >= MAIN:
            if time.time() - start > 0.1:
                time.sleep(0.01)
            continue

        start = time.time()
        if sem == RESET:
            seeds = recv_pipe.recv()
            _, infos = envs.reset(seed=seeds)
        elif sem == STEP:
            _, _, _, _, infos, _ = envs.step(atn_arr)
        elif sem == CLOSE:
            print("closing worker", worker_idx)
            send_pipe.send(None)
            break

        if infos:
            semaphores[worker_idx] = INFO
            send_pipe.send(infos)
        else:
            semaphores[worker_idx] = MAIN

class Multiprocessing:
    '''Runs environments in parallel using multiprocessing

    Use this vectorization module for most applications
    '''
    reset = reset
    step = step
    def __init__(self, env_creators, env_args, env_kwargs,
            num_envs, num_workers=None, batch_size=None, zero_copy=True, **kwargs):
        self.envs = [creator(*args, **kwargs) for (creator, args, kwargs)
            in zip(env_creators, env_args, env_kwargs)]
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs
        if zero_copy and num_envs % batch_size != 0:
            raise APIUsageError(
                'zero_copy: num_envs must be divisible by batch_size')

        self.num_envs = num_envs
        envs_per_worker = num_envs // num_workers
        self.envs_per_worker = envs_per_worker
        self.workers_per_batch = batch_size // envs_per_worker
        self.num_workers = num_workers

        # I really didn't want to need a driver process... with mp.shared_memory
        # we can fetch this data from the worker processes and ever perform
        # additional space checks. Unfortunately, SharedMemory has a janky integration
        # with the resource tracker that spams warnings and does not work with
        # forked processes. So for now, RawArray is much more reliable.
        # You can't send a RawArray through a pipe.
        driver_env = env_creators[0](*env_args[0], **env_kwargs[0])
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

        from multiprocessing import RawArray
        self.shm = namespace(
            observations=RawArray(obs_ctype, num_agents * int(np.prod(obs_shape))),
            actions=RawArray(atn_ctype, num_agents * int(np.prod(atn_shape))),
            rewards=RawArray('f', num_agents),
            terminals=RawArray('b', num_agents),
            truncateds=RawArray('b', num_agents),
            masks=RawArray('b', num_agents),
            semaphores=RawArray('c', num_workers),
        )
        shape = (num_workers, agents_per_worker)
        self.obs_batch_shape = (self.agents_per_batch, *obs_shape)
        self.atn_batch_shape = (self.workers_per_batch, agents_per_worker, *atn_shape)
        self.actions = np.ndarray((*shape, *atn_shape),
            dtype=atn_dtype, buffer=self.shm.actions)
        self.buf = namespace(
            observations=np.ndarray((*shape, *obs_shape),
                dtype=obs_dtype, buffer=self.shm.observations),
            rewards=np.ndarray(shape, dtype=np.float32, buffer=self.shm.rewards),
            terminals=np.ndarray(shape, dtype=bool, buffer=self.shm.terminals),
            truncations=np.ndarray(shape, dtype=bool, buffer=self.shm.truncateds),
            masks=np.ndarray(shape, dtype=bool, buffer=self.shm.masks),
            semaphores=np.ndarray(num_workers, dtype=np.uint8, buffer=self.shm.semaphores),
        )
        self.buf.semaphores[:] = MAIN

        from multiprocessing import Pipe, Process
        self.send_pipes, w_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        w_send_pipes, self.recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        self.recv_pipe_dict = {p: i for i, p in enumerate(self.recv_pipes)}

        self.processes = []
        for i in range(num_workers):
            start = i * envs_per_worker
            end = start + envs_per_worker
            p = Process(
                target=_worker_process,
                args=(env_creators[start:end], env_args[start:end],
                    env_kwargs[start:end], envs_per_worker,
                    num_workers, i, w_send_pipes[i], w_recv_pipes[i],
                    self.shm)
            )
            p.start()
            self.processes.append(p)

        self.flag = RESET
        self.zero_copy = zero_copy

    #@profile
    def recv(self):
        recv_precheck(self)
        while True:
            worker = self.waiting_workers.pop(0)
            sem = self.buf.semaphores[worker]
            if sem >= MAIN:
                self.ready_workers.append(worker)
            else:
                self.waiting_workers.append(worker)

            if sem == INFO:
                self.infos[worker] = self.recv_pipes[worker].recv()

            if not self.ready_workers:
                continue

            if self.workers_per_batch == 1:
                # Fastest path. Zero-copy optimized for batch size 1
                w_slice = self.ready_workers[0]
                s_range = [w_slice]
                self.waiting_workers.append(w_slice)
                self.ready_workers.pop(0)
                break
            elif self.zero_copy:
                # Zero-copy for batch size > 1. Has to wait for
                # a contiguous block of workers and adds a few
                # microseconds of extra index processing time
                completed = np.zeros(self.num_workers, dtype=bool)
                completed[self.ready_workers] = True
                buffers = completed.reshape(
                    -1, self.workers_per_batch).all(axis=1)
                start = buffers.argmax()
                if not buffers[start]:
                    continue

                start *= self.workers_per_batch
                end = start + self.workers_per_batch
                w_slice = slice(start, end)
                s_range = range(start, end)
                self.waiting_workers.extend(s_range)
                self.ready_workers = [e for e in self.ready_workers
                    if e not in s_range]
                break
            elif len(self.ready_workers) >= self.workers_per_batch:
                # Full async path for batch size > 1. Alawys copies
                # data because of non-contiguous worker indices
                # Can be faster for envs with small observations
                w_slice = self.ready_workers[:self.workers_per_batch]
                s_range = w_slice
                self.waiting_workers.extend(s_range)
                self.ready_workers = self.ready_workers[self.workers_per_batch:]
                break

        self.w_slice = w_slice
        buf = self.buf

        o = buf.observations[w_slice].reshape(self.obs_batch_shape)
        r = buf.rewards[w_slice].ravel()
        d = buf.terminals[w_slice].ravel()
        t = buf.truncations[w_slice].ravel()

        infos = []
        for i in s_range:
            if self.infos[i]:
                infos.extend(self.infos[i])
                self.infos[i] = []

        agent_ids = self.agent_ids[w_slice].ravel()
        m = buf.masks[w_slice].ravel()

        return o, r, d, t, infos, agent_ids, m

    #@profile
    def send(self, actions):
        send_precheck(self)
        actions = np.asarray(actions).reshape(self.atn_batch_shape)
        
        idxs = self.w_slice
        self.actions[idxs] = actions
        self.buf.semaphores[idxs] = STEP

    def async_reset(self, seed=42):
        seed = make_seeds(seed, self.num_envs)
        self.prev_env_id = []
        self.flag = RECV

        self.ready_workers = []
        self.waiting_workers = list(range(self.num_workers))
        self.infos = [[] for _ in range(self.num_workers)]

        self.buf.semaphores[:] = RESET
        for i in range(self.num_workers):
            start = i*self.envs_per_worker
            end = (i+1)*self.envs_per_worker
            self.send_pipes[i].send(seed[start:end])

    def close(self):
        for p in self.processes:
            p.terminate()

class Ray():
    '''Runs environments in parallel on multiple processes using Ray

    Use this module for distributed simulation on a cluster. It can also be
    faster than multiprocessing on a single machine for specific environments.
    '''
    reset = reset
    step = step
    def __init__(self, env_creators, env_args, env_kwargs,
            num_envs, num_workers=None, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs

        self.env_pool = num_envs != batch_size
        envs_per_worker = num_envs // num_workers
        self.workers_per_batch = batch_size // envs_per_worker
        self.num_workers = num_workers

        driver_env = env_creators[0](*env_args[0], **env_kwargs[0])
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

        self.envs = []
        for i in range(num_workers):
            start = i * envs_per_worker
            end = start + envs_per_worker
            self.envs.append(
                ray.remote(Serial).remote(
                    env_creators[start:end],
                    env_args[start:end],
                    env_kwargs[start:end],
                    envs_per_worker
                )
            )

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
        actions = np.asarray(actions).reshape(self.atn_batch_shape)
        handles = []
        for i, e in enumerate(self.prev_env_id):
            atns = actions[i]
            env = self.envs[e]
            env.send.remote(atns)
            handles.append(env.recv.remote())

        self.async_handles = handles

    def async_reset(self, seed=42):
        self.flag = RECV
        if seed is None:
            kwargs = {}
        else:
            kwargs = {"seed": seed}

        handles = []
        for idx, e in enumerate(self.envs):
            e.async_reset.remote(**kwargs)
            handles.append(e.recv.remote())

        self.async_handles = handles

    def close(self):
        self.ray.get([e.close.remote() for e in self.envs])
        self.ray.shutdown()


def make(env_creator_or_creators, env_args=None, env_kwargs=None, backend=Serial, num_envs=1, **kwargs):
    if num_envs < 1:
        raise APIUsageError('num_envs must be at least 1')
    if num_envs != int(num_envs):
        raise APIUsageError('num_envs must be an integer')
 
    if env_args is None:
        env_args = []

    if env_kwargs is None:
        env_kwargs = {}

    if not isinstance(env_creator_or_creators, (list, tuple)):
        env_creators = [env_creator_or_creators] * num_envs
        env_args = [env_args] * num_envs
        env_kwargs = [env_kwargs] * num_envs

    if len(env_creators) != num_envs:
        raise APIUsageError('env_creators must be a list of length num_envs')
    if len(env_args) != num_envs:
        raise APIUsageError('env_args must be a list of length num_envs')
    if len(env_kwargs) != num_envs:
        raise APIUsageError('env_kwargs must be a list of length num_envs')

    for i in range(num_envs):
        if not callable(env_creators[i]):
            raise APIUsageError('env_creators must be a list of callables')
        if not isinstance(env_args[i], (list, tuple)):
            raise APIUsageError('env_args must be a list of lists or tuples')
        if not isinstance(env_kwargs[i], dict):
            raise APIUsageError('env_kwargs must be a list of dictionaries')

    # Keeps batch size consistent when debugging with Serial backend
    if backend is Serial and 'batch_size' in kwargs:
        num_envs = kwargs['batch_size']

    # TODO: First step action space check
    
    return backend(env_creators, env_args, env_kwargs, num_envs, **kwargs)

def make_seeds(seed, num_envs):
    if isinstance(seed, int):
        return [seed + i for i in range(num_envs)]

    err = f'seed {seed} must be an integer or a list of integers'
    if isinstance(seed, (list, tuple)):
        if len(seed) != num_envs:
            raise APIUsageError(err)

        return seed

    raise APIUsageError(err)

def check_envs(envs, driver):
    valid = (PufferEnv, GymnasiumPufferEnv, PettingZooPufferEnv)
    driver_obs = driver.single_observation_space
    driver_atn = driver.single_action_space
    for env in envs:
        if not isinstance(env, valid):
            raise APIUsageError(f'env_creators must be {valid}')
        obs_space = env.single_observation_space
        if obs_space != driver_obs:
            raise APIUsageError(f'\n{obs_space}\n{driver_obs} obs space mismatch')
        atn_space = env.single_action_space
        if atn_space != driver_atn:
            raise APIUsageError(f'\n{atn_space}\n{driver_atn} atn space mismatch')
