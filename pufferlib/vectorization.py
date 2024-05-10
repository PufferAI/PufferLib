from pdb import set_trace as T
from collections.abc import Mapping

import numpy as np
import gymnasium
from itertools import chain
import psutil
import time
import msgpack


from pufferlib import namespace
from pufferlib.environment import PufferEnv
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.multi_env import PufferEnvWrapper
from pufferlib.exceptions import APIUsageError
import pufferlib.spaces

import pufferlib.exceptions
import pufferlib.emulation



RESET = 0
SEND = 1
RECV = 2

space_error_msg = 'env {env} must be an instance of GymnasiumPufferEnv or PettingZooPufferEnv'

def _single_observation_space(env):
    if isinstance(env, PufferEnv):
        return env.observation_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_observation_space
    elif isinstance(env, GymnasiumPufferEnv):
        return env.observation_space
    else:
        raise TypeError(space_error_msg.format(env=env))
 
def single_observation_space(state):
    return _single_observation_space(state.driver_env)

def _single_action_space(env):
    if isinstance(env, PufferEnv):
        return env.action_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_action_space
    elif isinstance(env, GymnasiumPufferEnv):
        return env.action_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_action_space(state):
    return _single_action_space(state.driver_env)

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

def buffer_scale(env_creator, env_args, env_kwargs, num_envs, num_workers, envs_per_batch=None):
    '''These calcs are simple but easy to mess up and hard to catch downstream.
    We do them all at once here to avoid that'''
    if num_envs % num_workers != 0:
        raise APIUsageError('num_envs must be divisible by num_workers')
    if num_envs < num_workers:
        raise APIUsageError('num_envs must be >= num_workers')
    if envs_per_batch is None:
        envs_per_batch = num_envs
    if envs_per_batch > num_envs:
        raise APIUsageError('envs_per_batch must be <= num_envs')
    if envs_per_batch % num_workers != 0:
        raise APIUsageError('envs_per_batch must be divisible by num_workers')
    if envs_per_batch < 1:
        raise APIUsageError('envs_per_batch must be > 0')

    if not callable(env_creator):
        raise pufferlib.exceptions.APIUsageError('env_creator must be callable')
    if not isinstance(env_args, (list, tuple)):
        raise pufferlib.exceptions.APIUsageError('env_args must be a list or tuple')
    # TODO: port namespace to Mapping
    if not isinstance(env_kwargs, Mapping):
        raise pufferlib.exceptions.APIUsageError('env_kwargs must be a dictionary or None')

    driver_env = env_creator(*env_args, **env_kwargs)

    if isinstance(driver_env, PufferEnv):
        agents_per_env = driver_env.num_agents
    elif isinstance(driver_env, PettingZooPufferEnv):
        agents_per_env = len(driver_env.agents)
    elif isinstance(driver_env, GymnasiumPufferEnv):
        agents_per_env = 1
    else:
        raise TypeError(
            'env_creator must return an instance '
            'of PufferEnv, GymnasiumPufferEnv or PettingZooPufferEnv'
        )

    num_agents = num_envs * agents_per_env

    envs_per_worker = num_envs // num_workers
    agents_per_worker = envs_per_worker * agents_per_env

    workers_per_batch = envs_per_batch // envs_per_worker
    agents_per_batch = envs_per_batch * agents_per_env

    observation_shape = _single_observation_space(driver_env).shape
    observation_dtype = _single_observation_space(driver_env).dtype
    action_shape = _single_action_space(driver_env).shape
    action_dtype = _single_action_space(driver_env).dtype

    observation_buffer_shape = (num_workers, agents_per_worker, *observation_shape)
    observation_batch_shape = (agents_per_batch, *observation_shape)
    action_buffer_shape = (num_workers, envs_per_worker, agents_per_env, *action_shape)
    action_batch_shape = (workers_per_batch, envs_per_worker, agents_per_env, *action_shape)
    batch_shape = (num_workers, agents_per_worker)

    agent_ids = np.stack([np.arange(
        i*agents_per_worker, (i+1)*agents_per_worker) for i in range(num_workers)])

    return driver_env, pufferlib.namespace(
        num_agents=num_agents,
        num_envs=num_envs,
        num_workers=num_workers,
        envs_per_batch=envs_per_batch,
        envs_per_worker=envs_per_worker,
        workers_per_batch=workers_per_batch,
        agents_per_batch=agents_per_batch,
        agents_per_worker=agents_per_worker,
        agents_per_env=agents_per_env,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        action_shape=action_shape,
        action_dtype=action_dtype,
        observation_buffer_shape=observation_buffer_shape,
        observation_batch_shape=observation_batch_shape,
        action_buffer_shape=action_buffer_shape,
        action_batch_shape=action_batch_shape,
        batch_shape=batch_shape,
        agent_ids=agent_ids,
    )


class Serial:
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)
 
    def __init__(self, env_creator: callable = None, env_args: list = [], env_kwargs: dict = {},
            num_envs: int = 1, num_workers: int = 1, envs_per_batch=None, mask_agents: bool = True, mem=None):
        if num_envs < 1:
            raise pufferlib.exceptions.APIUsageError('num_envs must be at least 1')

        self.driver_env = env_creator(*env_args, **env_kwargs)

        # Check that all envs are either Gymnasium or PettingZoo
        #is_gymnasium = all(isinstance(e, pufferlib.emulation.GymnasiumPufferEnv) for e in envs)
        #is_pettingzoo = all(isinstance(e, pufferlib.emulation.PettingZooPufferEnv) for e in envs)
        #is_puffer = all(isinstance(e, pufferlib.environment.PufferEnv) for e in envs)

        is_gymnasium = isinstance(self.driver_env, pufferlib.emulation.GymnasiumPufferEnv)
        is_pettingzoo = isinstance(self.driver_env, pufferlib.emulation.PettingZooPufferEnv)
        is_puffer = isinstance(self.driver_env, pufferlib.environment.PufferEnv)

        assert is_gymnasium or is_pettingzoo or is_puffer
        self.is_gymnasium = is_gymnasium
        self.is_pettingzoo = is_pettingzoo

        # Check that all envs have the same observation and action spaces
        # TODO: additional check here

        self.observation_space = self.driver_env.single_observation_space
        self.action_space = self.driver_env.single_action_space
        self.agents_per_env = self.driver_env.num_agents
        self.num_agents = self.agents_per_env * num_envs

        self.agents_per_batch = self.agents_per_env * num_envs #envs_per_batch
        self.agent_ids = np.arange(self.num_agents)

        if mem is None:
            #self.preallocated_obs = np.empty(
            #    (num_envs, self.agents_per_env), dtype=self.driver_env.emulated.emulated_observation_dtype)
            self.preallocated_obs = np.empty(
                (num_envs, self.agents_per_env, *self.observation_space.shape), dtype=self.observation_space.dtype)

            self.preallocated_rewards = np.empty((num_envs, self.agents_per_env), dtype=np.float32)
            self.preallocated_dones = np.empty((num_envs, self.agents_per_env), dtype=bool)
            self.preallocated_truncateds = np.empty((num_envs, self.agents_per_env), dtype=bool)
            self.preallocated_masks = np.ones((num_envs, self.agents_per_env), dtype=bool)

        else:
            self.preallocated_obs = mem.obs.reshape(num_envs, self.agents_per_env, *self.observation_space.shape)
            self.preallocated_rewards = mem.rew.reshape(num_envs, self.agents_per_env)
            self.preallocated_dones = mem.done.reshape(num_envs, self.agents_per_env)
            self.preallocated_truncateds = mem.trunc.reshape(num_envs, self.agents_per_env)
            self.preallocated_masks = mem.mask.reshape(num_envs, self.agents_per_env)

        self.return_obs = self.preallocated_obs.reshape(self.agents_per_batch, *self.observation_space.shape)
        self.return_rewards = self.preallocated_rewards.ravel()
        self.return_dones = self.preallocated_dones.ravel()
        self.return_truncateds = self.preallocated_truncateds.ravel()
        self.return_masks = self.preallocated_masks.ravel()

        self.action_batch_shape = (num_envs, self.agents_per_env, *self.driver_env.single_action_space.shape)

        envs = []
        for i in range(num_envs):
            mem = namespace(obs=self.preallocated_obs[i], rew=self.preallocated_rewards[i],
                done=self.preallocated_dones[i], trunc=self.preallocated_truncateds[i], mask=self.preallocated_masks[i])
            env = env_creator(*env_args, **env_kwargs)
            env.mem = mem
            envs.append(env)

        self.envs = envs

        #envs = [env_creator(*env_args, mem=mem, **env_kwargs) for _ in range(num_envs)]

        '''
        for idx, e in enumerate(envs):
            e.injected = True
            e.observations = self.preallocated_obs[idx].view(e.emulated.emulated_observation_dtype)
            e.rewards = self.preallocated_rewards[idx]
            e.dones = self.preallocated_dones[idx]
            e.truncateds = self.preallocated_truncateds[idx]
            e.masks = self.preallocated_masks[idx]
        '''


    def async_reset(self, seed=None):
        infos = []
        for idx, env in enumerate(self.envs):
            if seed is None:
                ob, i = env.reset()
            else:
                ob, i = env.reset(seed=hash(1000*seed + idx))
               
            infos.append(i)
            #self.preallocated_obs[idx] = ob

        self.preallocated_rewards[:] = 0
        self.preallocated_dones[:] = False
        self.preallocated_truncateds[:] = False
        self.preallocated_masks[:] = 1
        self.infos = infos

    def send(self, actions):
        rewards, dones, truncateds, self.infos = [], [], [], []

        actions = np.asarray(actions).reshape(self.action_batch_shape)
        for idx, env in enumerate(self.envs):
            atns = actions[idx]
            if env.done:
                o, i = env.reset()
                r = 0
                d = False
                t = False
            else:
                o, r, d, t, i = env.step(atns)

            self.infos.append(i)
            #self.preallocated_obs[idx] = o
            #self.preallocated_rewards[idx] = r
            #self.preallocated_dones[idx] = d
            #self.preallocated_truncateds[idx] = t

    def recv(self):
        return (self.return_obs, self.return_rewards, self.return_dones,
            self.return_truncateds, self.infos, self.agent_ids, self.return_masks)

        returns = [self.preallocated_obs, self.preallocated_rewards,
            self.preallocated_dones, self.preallocated_truncateds, self.infos]

        if self.mask_agents:
            returns.append(self.preallocated_masks)

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

def _worker_process(env_creator, env_args, env_kwargs,
        num_envs, agents_per_env, worker_idx, obs_shape, obs_mem, atn_shape, atn_mem, rewards_mem,
        terminals_mem, truncated_mem, mask_mem, observation_dtype, action_dtype, send_pipe, recv_pipe):
    
    # I don't know if this helps. Sometimes it does, sometimes not.
    # Need to run more comprehensive tests
    curr_process = psutil.Process()
    #curr_process.cpu_affinity([worker_idx+1])
    # Set to min niceness
    #curr_process.nice(19)


    num_agents = num_envs * agents_per_env
    obs_size = int(np.prod(obs_shape))
    obs_n = num_agents * obs_size

    atn_size = int(np.prod(atn_shape))
    atn_n = num_agents * atn_size

    s = worker_idx * num_agents
    e = (worker_idx + 1) * num_agents 
    s_obs = worker_idx * num_agents * obs_size
    e_obs = (worker_idx + 1) * num_agents * obs_size
    s_atn = worker_idx * num_agents * atn_size
    e_atn = (worker_idx + 1) * num_agents * atn_size

    obs_arr = np.frombuffer(obs_mem, dtype=observation_dtype)[s_obs:e_obs].reshape(num_agents, *obs_shape)
    atn_arr = np.frombuffer(atn_mem, dtype=action_dtype)[s_atn:e_atn]
    rewards_arr = np.frombuffer(rewards_mem, dtype=np.float32)[s:e]
    terminals_arr = np.frombuffer(terminals_mem, dtype=bool)[s:e]
    truncated_arr = np.frombuffer(truncated_mem, dtype=bool)[s:e]
    mask_arr = np.frombuffer(mask_mem, dtype=bool)[s:e]

    envs = Serial(env_creator, env_args, env_kwargs, num_envs,
        obs_mem=obs_arr, rew_mem=rewards_arr, done_mem=terminals_arr,
        trunc_mem=truncated_arr, mask_mem=mask_arr)

    while True:
        request = recv_pipe.recv_bytes()
        info = {}
        if request == RESET:
            response = envs.reset()
        elif request == STEP:
            response = envs.step(atn_arr.reshape(num_envs, agents_per_env, *atn_shape))

        send_pipe.send(envs.infos)

class Multiprocessing:
    '''Runs environments in parallel using multiprocessing

    Use this vectorization module for most applications
    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)

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

        self.num_agents = scale.num_agents
        self.agents_per_batch = scale.agents_per_batch

        # Shared memory for obs, rewards, terminals, truncateds
        from multiprocessing import Process, Manager, Pipe, Array
        from multiprocessing.sharedctypes import RawArray

        observation_size = int(np.prod(scale.observation_buffer_shape))
        action_size = int(np.prod(scale.action_buffer_shape))
        c_atn_dtype = np.ctypeslib.as_ctypes_type(scale.action_dtype)
        c_obs_dtype = np.ctypeslib.as_ctypes_type(scale.observation_dtype)
        batch_size = int(np.prod(scale.batch_shape))

        obs_mem = RawArray(c_obs_dtype, observation_size)
        atn_mem = RawArray(c_atn_dtype, action_size)
        rewards_mem = RawArray('f', batch_size)
        terminals_mem = RawArray('b', batch_size)
        truncated_mem = RawArray('b', batch_size)
        mask_mem = RawArray('b', batch_size)

        obs_arr = np.ndarray(scale.observation_buffer_shape, dtype=scale.observation_dtype, buffer=obs_mem)
        atn_arr = np.ndarray(scale.action_buffer_shape, dtype=scale.action_dtype, buffer=atn_mem)
        rewards_arr = np.ndarray(scale.batch_shape, dtype=np.float32, buffer=rewards_mem)
        terminals_arr = np.ndarray(scale.batch_shape, dtype=bool, buffer=terminals_mem)
        truncated_arr = np.ndarray(scale.batch_shape, dtype=bool, buffer=truncated_mem)
        mask_arr = np.ndarray(scale.batch_shape, dtype=bool, buffer=mask_mem)

        main_send_pipes, work_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        work_send_pipes, main_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        recv_pipe_dict = {p: i for i, p in enumerate(main_recv_pipes)}

        num_cores = psutil.cpu_count()
        processes = []
        for i in range(num_workers):
            p = Process(
                target=_worker_process,
                args=(env_creator, env_args, env_kwargs, scale.envs_per_worker, scale.agents_per_env, i,
                    scale.observation_shape, obs_mem, scale.action_shape, atn_mem, rewards_mem, terminals_mem, truncated_mem,
                    mask_mem, scale.observation_dtype, scale.action_dtype,
                    work_send_pipes[i], work_recv_pipes[i])
            )
            p.start()
            processes.append(p)

        # Register all receive pipes with the selector
        import selectors
        sel = selectors.DefaultSelector()
        for pipe in main_recv_pipes:
            sel.register(pipe, selectors.EVENT_READ)

        self.processes = processes
        self.sel = sel
        self.obs_arr = obs_arr
        self.atn_arr = atn_arr
        self.rewards_arr = rewards_arr
        self.terminals_arr = terminals_arr
        self.truncated_arr = truncated_arr
        self.mask_arr = mask_arr
        self.send_pipes = main_send_pipes
        self.recv_pipes = main_recv_pipes
        self.recv_pipe_dict = recv_pipe_dict
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.flag = RESET
        self.prev_env_id = []
        self.mask_agents = mask_agents
        self.env_pool = num_envs != envs_per_batch
        self.scale = scale

    def recv(self):
        recv_precheck(self)
        worker_ids = []
        infos = []
        if self.env_pool:
            while len(worker_ids) < self.scale.workers_per_batch:
                for key, _ in self.sel.select(timeout=None):
                    response_pipe = key.fileobj
                    info = response_pipe.recv()
                    infos.append(info)
                    env_id = self.recv_pipe_dict[response_pipe]
                    worker_ids.append(env_id)

                    if len(worker_ids) == self.scale.workers_per_batch:                    
                        break
        else:
            for env_id in range(self.scale.workers_per_batch):
                response_pipe = self.recv_pipes[env_id]
                info = response_pipe.recv()
                infos.append(info)
                worker_ids.append(env_id)

        infos = [i for ii in infos for i in ii]

        # Does not copy if workers_per_batch == 1
        if self.scale.workers_per_batch == 1:
            worker_ids = worker_ids[0]
        else:
            worker_ids = np.array(worker_ids)

        o = self.obs_arr[worker_ids].reshape(self.scale.observation_batch_shape)
        r = self.rewards_arr[worker_ids].ravel()
        d = self.terminals_arr[worker_ids].ravel()
        t = self.truncated_arr[worker_ids].ravel()
        m = self.mask_arr[worker_ids].ravel()

        self.prev_env_id = worker_ids
        agent_ids = self.scale.agent_ids[worker_ids].ravel()
        return o, r, d, t, infos, agent_ids, m

    def send(self, actions):
        send_precheck(self)
        actions = actions.reshape(self.scale.action_batch_shape)
        self.atn_arr[self.prev_env_id] = actions
        if self.scale.workers_per_batch == 1:
            self.send_pipes[self.prev_env_id].send_bytes(STEP)
        else:
            for i in self.prev_env_id:
                self.send_pipes[i].send_bytes(STEP)

    def async_reset(self, seed=None):
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
            pipe.send(("close", [], {}))

        for p in self.processes:
            p.terminate()

        for p in self.processes:
            p.join()

class Ray():
    '''Runs environments in parallel on multiple processes using Ray

    Use this module for distributed simulation on a cluster. It can also be
    faster than multiprocessing on a single machine for specific environments.
    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)

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
