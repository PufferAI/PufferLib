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
    if batch_size % num_workers != 0:
        raise pufferlib.exceptions.APIUsageError('batch_size must be divisible by num_workers')
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

def _worker_process(env_creator, env_args, env_kwargs, num_envs,
        num_workers, worker_idx, send_pipe, recv_pipe):
    
    envs = Serial(env_creator, env_args, env_kwargs, num_envs)
    num_agents = envs.num_agents
    obs_space = envs.single_observation_space
    obs_shape = envs.single_observation_space.shape
    obs_dtype = envs.single_observation_space.dtype
    atn_space = envs.single_action_space
    atn_shape = envs.single_action_space.shape
    atn_dtype = envs.single_action_space.dtype

    # Return shape and dtype information and wait for confirmation
    send_pipe.send((num_agents, obs_space, atn_space, envs.envs[0].emulated))
    shm = recv_pipe.recv()

    # Environments read and write directly to shared memory
    shape = (num_workers, num_agents)
    atn_arr = np.ndarray((*shape, *atn_shape), dtype=atn_dtype, buffer=shm.actions.buf)[worker_idx]
    buf = namespace(
        observations=np.ndarray((*shape, *obs_shape), dtype=obs_dtype, buffer=shm.observations.buf)[worker_idx],
        rewards=np.ndarray(shape, dtype=np.float32, buffer=shm.rewards.buf)[worker_idx],
        terminals=np.ndarray(shape, dtype=bool, buffer=shm.terminals.buf)[worker_idx],
        truncations=np.ndarray(shape, dtype=bool, buffer=shm.truncateds.buf)[worker_idx],
        masks=np.ndarray(shape, dtype=bool, buffer=shm.masks.buf)[worker_idx],
    )
    envs._assign_buffers(buf)

    while True:
        request = recv_pipe.recv_bytes()
        if request == RESET:
            response = envs.reset()
        elif request == STEP:
            response = envs.step(atn_arr)
        elif request == CLOSE:
            for e in shm.values():
                e.close()
            send_pipe.send(None)
            break

        send_pipe.send(envs.infos)

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
        self.env_pool = num_envs != batch_size
        envs_per_worker = num_envs // num_workers
        self.workers_per_batch = batch_size // envs_per_worker
        self.num_workers = num_workers

        import warnings
        warnings.filterwarnings("ignore", category=ResourceWarning)

        from multiprocessing import Pipe, Process, set_start_method
        self.send_pipes, w_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        w_send_pipes, self.recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        self.recv_pipe_dict = {p: i for i, p in enumerate(self.recv_pipes)}


        self.processes = []
        # Change start method to spawn for Windows compatibility
        #set_start_method('spawn', force=True)
        for i in range(num_workers):
            p = Process(
                target=_worker_process,
                args=(env_creator, env_args, env_kwargs, envs_per_worker,
                    num_workers, i, w_send_pipes[i], w_recv_pipes[i])
            )
            p.start()
            self.processes.append(p)

        # Check that all environments have the same observation and action spaces
        # When using envpool (batch_size < num_envs), all workers must have the same number of agents
        w_agents, w_obs_space, w_atn_space, w_emulated = zip(*[pipe.recv() for pipe in self.recv_pipes])
        assert all(o == w_obs_space[0] for o in w_obs_space)
        assert all(a == w_atn_space[0] for a in w_atn_space)
        assert all(e == w_emulated[0] for e in w_emulated)
        self.num_agents = num_agents = sum(w_agents)
        agents_per_worker = w_agents[0]
        self.agents_per_batch = self.workers_per_batch * agents_per_worker
        self.emulated = w_emulated[0]
        if batch_size < num_envs:
            assert all(w == agents_per_worker for w in w_agents)

        # TODO: Var agent envs
        self.agent_ids = np.arange(num_agents).reshape(num_workers, w_agents[0])

        obs_space = w_obs_space[0]
        obs_shape = obs_space.shape
        obs_dtype = obs_space.dtype
        self.single_observation_space = obs_space

        atn_space = w_atn_space[0]
        atn_shape = atn_space.shape
        atn_dtype = atn_space.dtype
        self.single_action_space = atn_space

        '''
        from multiprocessing.managers import SharedMemoryManager
        self.smm = SharedMemoryManager()
        self.smm.start()
        self.obs_mem = self.smm.SharedMemory(obs_dtype.itemsize * num_agents * prod(obs_shape))
        self.atn_mem = self.smm.SharedMemory(atn_dtype.itemsize * num_agents * prod(atn_shape))
        self.rewards_mem = self.smm.SharedMemory(4 * num_agents)
        self.terminals_mem = self.smm.SharedMemory(num_agents)
        self.truncated_mem = self.smm.SharedMemory(num_agents)
        self.mask_mem = self.smm.SharedMemory(num_agents)
        '''
        from pufferlib.shared_memory import SharedMemory
        self.shm = namespace(
            observations=SharedMemory(create=True, size=obs_dtype.itemsize * num_agents * prod(obs_shape)),
            actions=SharedMemory(create=True, size=atn_dtype.itemsize * num_agents * prod(atn_shape)),
            rewards=SharedMemory(create=True, size=4 * num_agents),
            terminals=SharedMemory(create=True, size=num_agents),
            truncateds=SharedMemory(create=True, size=num_agents),
            masks=SharedMemory(create=True, size=num_agents),
        )

        shape = (num_workers, agents_per_worker)
        self.obs_batch_shape = (self.workers_per_batch*agents_per_worker, *obs_shape)
        self.atn_batch_shape = (self.workers_per_batch, agents_per_worker, *atn_shape)
        self.actions = np.ndarray((*shape, *atn_shape), dtype=atn_dtype, buffer=self.shm.actions.buf)
        self.buf = namespace(
            observations=np.ndarray((*shape, *obs_shape), dtype=obs_dtype, buffer=self.shm.observations.buf),
            rewards=np.ndarray(shape, dtype=np.float32, buffer=self.shm.rewards.buf),
            terminals=np.ndarray(shape, dtype=bool, buffer=self.shm.terminals.buf),
            truncations=np.ndarray(shape, dtype=bool, buffer=self.shm.truncateds.buf),
            masks=np.ndarray(shape, dtype=bool, buffer=self.shm.masks.buf),
        )

        # Send confirmation to workers
        for pipe in self.send_pipes:
            pipe.send(self.shm)

        # Register all receive pipes with the selector
        import selectors
        self.sel = selectors.DefaultSelector()
        for pipe in self.recv_pipes:
            self.sel.register(pipe, selectors.EVENT_READ)

        self.flag = RESET

    def recv(self):
        recv_precheck(self)
        worker_ids = []
        infos = []
        if self.env_pool:
            while len(worker_ids) < self.workers_per_batch:
                for key, _ in self.sel.select(timeout=None):
                    response_pipe = key.fileobj
                    info = response_pipe.recv()
                    infos.append(info)
                    env_id = self.recv_pipe_dict[response_pipe]
                    worker_ids.append(env_id)

                    if len(worker_ids) == self.workers_per_batch:                    
                        break

            if self.workers_per_batch == 1:
                idxs = worker_ids[0]
            else:
                idxs = np.array(worker_ids)
     
        else:
            for env_id in range(self.num_workers):
                response_pipe = self.recv_pipes[env_id]
                info = response_pipe.recv()
                infos.append(info)

            idxs = ()


        buf = self.buf
        o = buf.observations[idxs].reshape(self.obs_batch_shape)
        r = buf.rewards[idxs].ravel()
        d = buf.terminals[idxs].ravel()
        t = buf.truncations[idxs].ravel()
        infos = [i for ii in infos for i in ii]
        agent_ids = self.agent_ids[idxs].ravel()
        m = buf.masks[idxs].ravel()

        self.prev_env_id = idxs
        return o, r, d, t, infos, agent_ids, m

    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.atn_batch_shape)
        
        if self.env_pool:
            self.actions[self.prev_env_id] = actions
            idxs = self.prev_env_id
        else:
            self.actions[:] = actions
            idxs = range(self.num_workers)

        for i in idxs:
            self.send_pipes[i].send_bytes(STEP)

    def async_reset(self, seed=None):
        self.prev_env_id = []
        reset_precheck(self)
        for pipe in self.send_pipes:
            pipe.send_bytes(RESET)

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
        for pipe in self.send_pipes:
            pipe.send_bytes(CLOSE)

        for pipe in self.recv_pipes:
            pipe.recv()

        for e in self.shm.values():
            e.close()
            e.unlink()

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
            num_workers: int = 1,
            envs_per_batch: int = None,
            mask_agents: bool = False,
            ) -> None:

        self.driver_env, scale = buffer_scale(env_creator, env_args, env_kwargs,
            num_envs, num_workers, envs_per_batch=None)

        import ray
        if not ray.is_initialized():
            import logging
            ray.init(
                include_dashboard=False,  # WSL Compatibility
                logging_level=logging.ERROR,
            )

        multi_envs = [
            ray.remote(PufferEnvWrapper).remote(
                env_creator, env_args, env_kwargs, scale.envs_per_worker
            ) for _ in range(num_workers)
        ]

        self.multi_envs = multi_envs
        self.async_handles = None
        self.flag = RESET
        self.ray = ray
        self.prev_env_id = []
        self.env_pool = num_envs != envs_per_batch
        self.mask_agents = mask_agents
        self.scale = scale

    def recv(self):
        recv_precheck(self)
        recvs = []
        next_env_id = []
        workers_per_batch = self.scale.workers_per_batch
        if self.env_pool:
            recvs = self.ray.get(self.async_handles[:workers_per_batch])
            env_id = [_ for _ in range(workers_per_batch)]
        else:
            ready, busy = self.ray.wait(
                self.async_handles, num_returns=workers_per_batch)
            env_id = [self.async_handles.index(e) for e in ready]
            recvs = self.ray.get(ready)

        
        o, r, d, t, infos, m = zip(*recvs)
        self.prev_env_id = env_id

        infos = [i for ii in infos for i in ii]

        o = np.stack(o, axis=0).reshape(self.scale.observation_batch_shape)
        r = np.stack(r, axis=0).ravel()
        d = np.stack(d, axis=0).ravel()
        t = np.stack(t, axis=0).ravel()
        m = np.stack(m, axis=0).ravel()
        agent_ids = self.scale.agent_ids[env_id].ravel()
        return o, r, d, t, infos, agent_ids, m

    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.scale.action_batch_shape)
        handles = []
        for i, e in enumerate(self.prev_env_id):
            atns = actions[i]
            handles.append(self.multi_envs[e].step.remote(atns))

        self.async_handles = handles

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            kwargs = {}
        else:
            kwargs = {"seed": seed}

        handles = []
        for idx, e in enumerate(self.multi_envs):
            handles.append(e.reset.remote(**kwargs))

        self.async_handles = handles

    def put(self, *args, **kwargs):
        for e in self.multi_envs:
            e.put.remote(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.ray.get([e.get.remote(*args, **kwargs) for e in self.multi_envs])

    def close(self):
        self.ray.get([e.close.remote() for e in self.multi_envs])
        self.ray.shutdown()
