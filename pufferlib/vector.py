# TODO: Check actions passed to envs are right shape? On first call at least

from pdb import set_trace as T

import numpy as np
import time
import psutil

from pufferlib import namespace
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.environment import PufferEnv
from pufferlib.exceptions import APIUsageError
from pufferlib.namespace import Namespace
import pufferlib.spaces
import gymnasium

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

def send_precheck(vecenv, actions):
    if vecenv.flag != SEND:
        raise APIUsageError('Call (async) reset + recv before sending')

    actions = np.asarray(actions)
    if not vecenv.initialized:
        vecenv.initialized = True
        if not vecenv.action_space.contains(actions):
            raise APIUsageError('Actions do not match action space')

    vecenv.flag = RECV
    return actions

def reset(vecenv, seed=42):
    vecenv.async_reset(seed)
    obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
    return obs, infos

def step(vecenv, actions):
    actions = np.asarray(actions)
    vecenv.send(actions)
    obs, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()
    return obs, rewards, terminals, truncations, infos # include env_ids or no?

class Native:
    reset = reset
    step = step

    @property
    def emulated(self):
        '''Native envs do not use emulation'''
        return False

    @property
    def done(self):
        '''Native envs handle resets internally'''
        return False

    @property
    def driver_env(self):
        return self.env

    @property
    def num_agents(self):
        return self.env.num_agents

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def single_observation_space(self):
        return self.env.single_observation_space

    @property
    def single_action_space(self):
        return self.env.single_action_space

    def __init__(self, env_creator, env_args, env_kwargs, buf=None, **kwargs):
        self.env = env_creator(*env_args, buf=buf, **env_kwargs)
        assert isinstance(self.env, PufferEnv)

        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.num_agents)
        self.agent_ids = np.arange(self.num_agents)
        self.flag = RESET

        if not isinstance(self.single_observation_space, pufferlib.spaces.Box):
            raise APIUsageError('Native observation_space must be a Box')
        if (not isinstance(self.single_action_space, pufferlib.spaces.Discrete)
                and not isinstance(self.single_action_space, pufferlib.spaces.MultiDiscrete)
                and not isinstance(self.single_action_space, pufferlib.spaces.Box)):
            raise APIUsageError('Native action_space must be a Discrete, MultiDiscrete, or Box')

    def async_reset(self, seed=None):
        self.flag = RECV
        _, self.infos = self.env.reset(seed)
        assert isinstance(self.infos, list), 'PufferEnvs must return info as a list of dicts'

    def send(self, actions):
        _, _, _, _, self.infos = self.env.step(actions)
        assert isinstance(self.infos, list), 'PufferEnvs must return info as a list of dicts'

    def recv(self):
        return (self.env.observations, self.env.rewards, self.env.terminals,
            self.env.truncations, self.infos, self.agent_ids, self.env.masks)

    def close(self):
        self.env.close()
 
class Serial:
    reset = reset
    step = step

    @property
    def num_envs(self):
        return self.agents_per_batch
 
    def __init__(self, env_creators, env_args, env_kwargs, num_envs, **kwargs):
        self.envs = [creator(*args, **kwargs) for (creator, args, kwargs)
            in zip(env_creators, env_args, env_kwargs)]

        self.driver_env = driver = self.envs[0]
        self.emulated = self.driver_env.emulated
        check_envs(self.envs, self.driver_env)
        self.agents_per_env = [env.num_agents for env in self.envs]
        self.agents_per_batch = sum(self.agents_per_env)
        self.num_agents = sum(self.agents_per_env)
        self.single_observation_space = driver.single_observation_space
        self.single_action_space = driver.single_action_space
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.agents_per_batch)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.agents_per_batch)
        self.agent_ids = np.arange(self.num_agents)
        self.initialized = False
        self.flag = RESET
        self.buf = None

    def _assign_buffers(self, buf):
        '''Envs handle their own data buffers'''
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
        self.flag = RECV
        seed = make_seeds(seed, len(self.envs))

        if self.buf is None:
            self.buf = namespace(
                observations = np.zeros(
                    (self.agents_per_batch, *self.single_observation_space.shape),
                    dtype=self.single_observation_space.dtype),
                rewards = np.zeros(self.agents_per_batch, dtype=np.float32),
                terminals = np.zeros(self.agents_per_batch, dtype=bool),
                truncations = np.zeros(self.agents_per_batch, dtype=bool),
                masks = np.ones(self.agents_per_batch, dtype=bool),
            )
            self._assign_buffers(self.buf)

        infos = []
        for env, s in zip(self.envs, seed):
            ob, i = env.reset(seed=s)
               
            if i:
                infos.append(i)

        self.infos = infos

    def send(self, actions):
        if not actions.flags.contiguous:
            actions = np.ascontiguousarray(actions)

        actions = send_precheck(self, actions)
        rewards, dones, truncateds, self.infos = [], [], [], []
        ptr = 0
        for idx, env in enumerate(self.envs):
            end = ptr + self.agents_per_env[idx]
            atns = actions[ptr:end]
            if env.done:
                o, i = env.reset()
                buf = self.buf
            else:
                o, r, d, t, i = env.step(atns)

            if i:
                self.infos.append(i)

            ptr = end

    def recv(self):
        recv_precheck(self)
        buf = self.buf
        return (buf.observations, buf.rewards, buf.terminals, buf.truncations,
            self.infos, self.agent_ids, buf.masks)

    def close(self):
        for env in self.envs:
            env.close()

def _worker_process(env_creators, env_args, env_kwargs, obs_shape, obs_dtype, atn_shape, atn_dtype,
        num_envs, num_agents, num_workers, worker_idx, send_pipe, recv_pipe, shm, is_native):

    # Environments read and write directly to shared memory
    shape = (num_workers, num_agents)
    atn_arr = np.ndarray((*shape, *atn_shape),
        dtype=atn_dtype, buffer=shm.actions)[worker_idx]
    buf = namespace(
        observations=np.ndarray((*shape, *obs_shape),
            dtype=obs_dtype, buffer=shm.observations)[worker_idx],
        rewards=np.ndarray(shape, dtype=np.float32, buffer=shm.rewards)[worker_idx],
        terminals=np.ndarray(shape, dtype=bool, buffer=shm.terminals)[worker_idx],
        truncations=np.ndarray(shape, dtype=bool, buffer=shm.truncateds)[worker_idx],
        masks=np.ndarray(shape, dtype=bool, buffer=shm.masks)[worker_idx],
        actions=atn_arr,
    )
    buf.masks[:] = True

    if is_native:
        envs = Native(env_creators[0], env_args[0], env_kwargs[0], buf=buf)
    else:
        envs = Serial(env_creators, env_args, env_kwargs, num_envs)
        envs._assign_buffers(buf)

    semaphores=np.ndarray(num_workers, dtype=np.uint8, buffer=shm.semaphores)
    start = time.time()
    while True:
        sem = semaphores[worker_idx]
        if sem >= MAIN:
            if time.time() - start > 0.5:
                time.sleep(0.01)
            continue

        start = time.time()
        if sem == RESET:
            seeds = recv_pipe.recv()
            _, infos = envs.reset(seed=seeds)
        elif sem == STEP:
            _, _, _, _, infos = envs.step(atn_arr)
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

    @property
    def num_envs(self):
        return self.agents_per_batch
 
    def __init__(self, env_creators, env_args, env_kwargs,
            num_envs, num_workers=None, batch_size=None,
            zero_copy=True, **kwargs):
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs

        num_batches = num_envs / batch_size
        if zero_copy and num_batches != int(num_batches):
            # This is so you can have n equal buffers
            raise APIUsageError(
                'zero_copy: num_envs must be divisible by batch_size')

        self.num_environments = num_envs
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
        self.driver_env = driver_env = env_creators[0](*env_args[0], **env_kwargs[0])
        is_native = isinstance(driver_env, PufferEnv)
        if is_native and envs_per_worker != 1:
            raise APIUsageError('Native PufferEnvs should run multiple envs internally, not in Multiprocessing')
        self.emulated = False if is_native else driver_env.emulated
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
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.agents_per_batch)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.agents_per_batch)
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
                    env_kwargs[start:end], obs_shape, obs_dtype,
                    atn_shape, atn_dtype, envs_per_worker, driver_env.num_agents,
                    num_workers, i, w_send_pipes[i], w_recv_pipes[i],
                    self.shm, is_native)
            )
            p.start()
            self.processes.append(p)

        self.flag = RESET
        self.initialized = False
        self.zero_copy = zero_copy

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
            elif self.workers_per_batch == self.num_workers:
                # Slowest path. Zero-copy synchornized for all workers
                if len(self.ready_workers) < self.num_workers:
                    continue

                w_slice = slice(0, self.num_workers)
                s_range = range(0, self.num_workers)
                self.waiting_workers.extend(s_range)
                self.ready_workers = []
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
        self.batch_mask = m

        return o, r, d, t, infos, agent_ids, m

    def send(self, actions):
        actions = send_precheck(self, actions).reshape(self.atn_batch_shape)
        # TODO: What shape?
        
        idxs = self.w_slice
        self.actions[idxs] = actions
        self.buf.semaphores[idxs] = STEP

    def async_reset(self, seed=42):
        self.flag = RECV
        seed = make_seeds(seed, self.num_environments)
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

    Use this module for distributed simulation on a cluster.
    '''
    reset = reset
    step = step

    def __init__(self, env_creators, env_args, env_kwargs, num_envs,
            num_workers=None, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = num_envs
        if num_workers is None:
            num_workers = num_envs

        self.env_pool = num_envs != batch_size
        envs_per_worker = num_envs // num_workers
        self.workers_per_batch = batch_size // envs_per_worker
        self.num_workers = num_workers

        driver_env = env_creators[0](*env_args[0], **env_kwargs[0])
        self.driver_env = driver_env
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
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.agents_per_batch)
        self.observation_space = pufferlib.spaces.joint_space(self.single_observation_space, self.agents_per_batch)
 
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
        self.initialized = False
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
        actions = send_precheck(self, actions).reshape(self.atn_batch_shape)
        # TODO: What shape?

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

    if backend is Native:
        env_args = env_args or []
        env_kwargs = env_kwargs or {}
        return env_creator_or_creators(*env_args, **env_kwargs)
        #return Native(env_creator_or_creators, env_args, env_kwargs, **kwargs)

    if 'num_workers' in kwargs:
        num_workers = kwargs['num_workers']
        # TODO: None?
        envs_per_worker = num_envs / num_workers
        if envs_per_worker != int(envs_per_worker):
            raise APIUsageError('num_envs must be divisible by num_workers')

        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
            if batch_size is None:
                batch_size = num_envs

            if batch_size % envs_per_worker != 0:
                raise APIUsageError(
                    'batch_size must be divisible by (num_envs / num_workers)')
        
 
    if env_args is None:
        env_args = []

    if env_kwargs is None:
        env_kwargs = {}

    if not isinstance(env_creator_or_creators, (list, tuple)):
        env_creators = [env_creator_or_creators] * num_envs
        env_args = [env_args] * num_envs
        env_kwargs = [env_kwargs] * num_envs
    else:
        env_creators = env_creator_or_creators

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
        if not isinstance(env_kwargs[i], (dict, Namespace)):
            raise APIUsageError('env_kwargs must be a list of dictionaries')

    # Keeps batch size consistent when debugging with Serial backend
    if backend is Serial and 'batch_size' in kwargs:
        num_envs = kwargs['batch_size']

    # TODO: Check num workers is not greater than num envs. This results in
    # different Serial vs Multiprocessing behavior

    # Sanity check args
    for k in kwargs:
        if k not in ['num_workers', 'batch_size', 'zero_copy','backend']:
            raise APIUsageError(f'Invalid argument: {k}')

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
    if not isinstance(driver, valid):
        raise APIUsageError(f'env_creator must be {valid}')

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

def autotune(env_creator, batch_size, max_envs=194, model_forward_s=0.0,
        max_env_ram_gb=32, max_batch_vram_gb=0.05, time_per_test=5): 
    '''Determine the optimal vectorization parameters for your system'''
    # TODO: fix multiagent

    if batch_size is None:
        raise ValueError('batch_size must not be None')

    if max_envs < batch_size:
        raise ValueError('max_envs < min_batch_size')

    num_cores = psutil.cpu_count(logical=False)
    idle_ram = psutil.Process().memory_info().rss
    load_ram = idle_ram

    # Initial profile to estimate single-core performance
    print('Profiling single-core performance for ~', time_per_test, 'seconds')
    env = env_creator()
    env.reset()
    obs_space = env.single_observation_space
    actions = [
        np.array([env.single_action_space.sample()
            for _ in range(env.num_agents)])
        for _ in range(1000)
    ]

    num_agents = env.num_agents
    steps = 0
    step_times = []
    reset_times = []
    start = time.time()
    while time.time() - start < time_per_test:
        idle_ram = max(idle_ram, psutil.Process().memory_info().rss)
        s = time.time()
        if env.done:
            env.reset()
            reset_times.append(time.time() - s)
        else:
            env.step(actions[steps%1000])
            step_times.append(time.time() - s)
        steps += 1

    env.close()
    sum_time = sum(step_times) + sum(reset_times)
    reset_percent = 100 * sum(reset_times) / sum_time
    sps = steps * num_agents / sum_time
    step_variance = 100 * np.std(step_times) / np.mean(step_times)
    reset_mean = np.mean(reset_times)
    ram_usage = max(1, (idle_ram - load_ram)) / 1e9

    obs_size_gb = (
        np.prod(obs_space.shape)
        * np.dtype(obs_space.dtype).itemsize
        * num_agents
        / 1e9
    )

    # Max bandwidth
    bandwidth = obs_size_gb * sps
    throughput = bandwidth * num_cores

    print('Profile complete')
    print(f'    SPS: {sps:.3f}')
    print(f'    STD: {step_variance:.3f}%')
    print(f'    Reset: {reset_percent:.3f}%')
    print(f'    RAM: {1000*ram_usage:.3f} MB/env')
    print(f'    Bandwidth: {bandwidth:.3f} GB/s')
    print(f'    Throughput: {throughput:.3f} GB/s ({num_cores} cores)')
    print()

    # Cap envs based on max allowed RAM
    max_allowed_by_ram = max_env_ram_gb // ram_usage
    if max_allowed_by_ram < max_envs:
        max_envs = int(max_allowed_by_ram)
        print('Reducing max envs to', max_envs, 'based on RAM')

    # Cap envs based on estimated max speedup
    #linear_speedup = (num_cores * steps / sum_time) // 500
    #if linear_speedup < max_envs and linear_speedup > num_cores:
    #    max_envs = int(linear_speedup)
    #    print('Reducing max envs to', max_envs, 'based on single-core speed')

    # Cap envs based on hardware
    hardware_envs = max_envs - (max_envs % num_cores)
    if hardware_envs > batch_size and hardware_envs != max_envs:
        max_envs = int(hardware_envs)
        print('Reducing max envs to', max_envs, 'based on core division')

    max_allowed_by_vram = max_batch_vram_gb // obs_size_gb
    if max_allowed_by_vram < batch_size:
        raise ValueError('max_allowed_by_vram < batch_size')

    print()
    configs = []

    # Strategy 1: one batch per core
    strategy_cores = min(num_cores, max_envs // batch_size)
    configs.append(dict(
        num_envs=batch_size*strategy_cores,
        num_workers=strategy_cores,
        batch_size=batch_size,
        backend=Multiprocessing,
    ))

    strategy_min_envs_per_worker = int(np.ceil((batch_size+1) / num_cores))
    strategy_num_envs = []
    for envs_per_worker in range(strategy_min_envs_per_worker, batch_size):
        num_envs = envs_per_worker * num_cores
        if num_envs > max_envs:
            break
        elif batch_size % envs_per_worker != 0:
            continue

        # Strategy 2: Full async. Only reasonable for low bandwidth
        #if throughput < 1.5:
        configs.append(dict(
            num_envs=num_envs,
            num_workers=num_cores,
            batch_size=batch_size,
            zero_copy=False,
            backend=Multiprocessing,
        ))

        # Strategy 3: Contiguous blocks. Only reasonable for high bandwidth
        num_batchs = num_envs / batch_size
        if num_batchs != int(num_batchs):
            continue
        if throughput > 0.5:
            configs.append(dict(
                num_envs=num_envs,
                num_workers=num_cores,
                batch_size=batch_size,
                backend=Multiprocessing,
            ))
        
    # Strategy 4: Full sync - perhaps nichely useful
    for strategy_cores in range(num_cores, 1, -1):
        if batch_size % strategy_cores != 0:
            continue

        configs.append(dict(
            num_envs=batch_size,
            num_workers=strategy_cores,
            batch_size=batch_size,
            backend=Multiprocessing,
        ))

    # Strategy 5: Serial
    configs.append(dict(
        num_envs=batch_size,
        backend=Serial,
    ))

    for config in configs:
        with pufferlib.utils.Suppress():
            envs = make(env_creator, **config)
            envs.reset()
        actions = [envs.action_space.sample() for _ in range(1000)]
        step_time = 0
        steps = 0
        start = time.time()
        while time.time() - start < time_per_test:
            s = time.time()
            envs.send(actions[steps%1000])
            step_time += time.time() - s

            if model_forward_s > 0:
                time.sleep(model_forward_s)

            s = time.time()
            envs.recv()
            step_time += time.time() - s

            steps += 1

        end = time.time()
        envs.close()
        sps = steps * envs.agents_per_batch / step_time
        print(f'SPS: {sps:.3f}')
        for k, v in config.items():
            if k == 'backend':
                v = v.__name__

            print(f'    {k}: {v}')

        print()
