from pdb import set_trace as T

import numpy as np
import gymnasium
from itertools import chain
import psutil
import time


from pufferlib import namespace
from pufferlib.emulation import GymnasiumPufferEnv, PettingZooPufferEnv
from pufferlib.multi_env import create_precheck, GymnasiumMultiEnv, PettingZooMultiEnv
from pufferlib.exceptions import APIUsageError
import pufferlib.spaces


RESET = 0
SEND = 1
RECV = 2

space_error_msg = 'env {env} must be an instance of GymnasiumPufferEnv or PettingZooPufferEnv'


def calc_scale_params(num_envs, envs_per_batch, envs_per_worker, agents_per_env):
    '''These calcs are simple but easy to mess up and hard to catch downstream.
    We do them all at once here to avoid that'''

    if num_envs % envs_per_worker != 0:
        raise APIUsageError('num_envs must be divisible by envs_per_worker')
    
    num_workers = num_envs // envs_per_worker
    envs_per_batch = num_envs if envs_per_batch is None else envs_per_batch

    if envs_per_batch > num_envs:
        raise APIUsageError('envs_per_batch must be <= num_envs')
    if envs_per_batch % envs_per_worker != 0:
        raise APIUsageError('envs_per_batch must be divisible by envs_per_worker')
    if envs_per_batch < 1:
        raise APIUsageError('envs_per_batch must be > 0')

    workers_per_batch = envs_per_batch // envs_per_worker
    assert workers_per_batch <= num_workers

    agents_per_batch = envs_per_batch * agents_per_env
    agents_per_worker = envs_per_worker * agents_per_env
 
    return num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker

def setup(env_creator, env_args, env_kwargs):
    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    driver_env = env_creator(*env_args, **env_kwargs)

    if isinstance(driver_env, GymnasiumPufferEnv):
        multi_env_cls = GymnasiumMultiEnv 
        env_agents = 1
        is_multiagent = False
    elif isinstance(driver_env, PettingZooPufferEnv):
        multi_env_cls = PettingZooMultiEnv
        env_agents = len(driver_env.possible_agents)
        is_multiagent = True
    else:
        raise TypeError(
            'env_creator must return an instance '
            'of GymnasiumPufferEnv or PettingZooPufferEnv'
        )

    obs_space = _single_observation_space(driver_env)
    return driver_env, multi_env_cls, env_agents

def _single_observation_space(env):
    if isinstance(env, GymnasiumPufferEnv):
        return env.observation_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_observation_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_observation_space(state):
    return _single_observation_space(state.driver_env)

def _single_action_space(env):
    if isinstance(env, GymnasiumPufferEnv):
        return env.action_space
    elif isinstance(env, PettingZooPufferEnv):
        return env.single_action_space
    else:
        raise TypeError(space_error_msg.format(env=env))

def single_action_space(state):
    return _single_action_space(state.driver_env)

def structured_observation_space(state):
    return state.driver_env.structured_observation_space

def flat_observation_space(state):
    return state.driver_env.flat_observation_space

def unpack_batched_obs(state, obs):
    return state.driver_env.unpack_batched_obs(obs)

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
    self.send(actions)
    return self.recv()[:-1]

def aggregate_recvs(state, recvs):
    obs, rewards, dones, truncateds, infos, env_ids = list(zip(*recvs))
    assert all(state.workers_per_batch == len(e) for e in
        (obs, rewards, dones, truncateds, infos, env_ids))

    obs = np.concatenate(obs)
    rewards = np.concatenate(rewards)
    dones = np.concatenate(dones)
    truncateds = np.concatenate(truncateds)
    infos = [i for ii in infos for i in ii]
    
    obs_space = state.driver_env.structured_observation_space
    if isinstance(obs_space, pufferlib.spaces.Box):
        obs = obs.reshape(obs.shape[0], *obs_space.shape)

    # TODO: Masking will break for 1-agent PZ envs
    # Replace with check against is_multiagent (add it to state)
    if state.agents_per_env > 1:
        mask = [e['mask'] for ee in infos for e in ee.values()]
    else:
        mask = [e['mask'] for e in infos]

    env_ids = np.concatenate([np.arange( # Per-agent env indexing
        i*state.agents_per_worker, (i+1)*state.agents_per_worker) for i in env_ids])

    assert all(state.agents_per_batch == len(e) for e in
        (obs, rewards, dones, truncateds, env_ids, mask))
    assert len(infos) == state.envs_per_batch

    if state.mask_agents:
        return obs, rewards, dones, truncateds, infos, env_ids, mask

    return obs, rewards, dones, truncateds, infos, env_ids

def split_actions(state, actions, env_id=None):
    assert isinstance(actions, (list, np.ndarray))
    if type(actions) == list:
        actions = np.array(actions)

    assert len(actions) == state.agents_per_batch
    return np.array_split(actions, state.workers_per_batch)


class Serial:
    '''Runs environments in serial on the main process
    
    Use this vectorization module for debugging environments
    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)
    structured_observation_space = property(structured_observation_space)
    flat_observation_space = property(flat_observation_space)
    unpack_batched_obs = unpack_batched_obs
    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            num_envs: int = 1,
            envs_per_worker: int = 1,
            envs_per_batch: int = None,
            env_pool: bool = False,
            mask_agents: bool = False,
            ) -> None:
        self.driver_env, self.multi_env_cls, self.agents_per_env = setup(
            env_creator, env_args, env_kwargs)

        self.num_envs = num_envs
        self.num_workers, self.workers_per_batch, self.envs_per_batch, self.agents_per_batch, self.agents_per_worker = calc_scale_params(
            num_envs, envs_per_batch, envs_per_worker, self.agents_per_env)
        self.envs_per_worker = envs_per_worker
        self.mask_agents = mask_agents

        self.multi_envs = [
            self.multi_env_cls(
                env_creator, env_args, env_kwargs, envs_per_worker,
            ) for _ in range(self.num_workers)
        ]

        self.flag = RESET

    def recv(self):
        recv_precheck(self)
        recvs = [(o, r, d, t, i, env_id) for (o, r, d, t, i), env_id
            in zip(self.data, range(self.workers_per_batch))]
        return aggregate_recvs(self, recvs)

    def send(self, actions):
        send_precheck(self)
        actions = split_actions(self, actions)
        self.data = [e.step(a) for e, a in zip(self.multi_envs, actions)]

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            self.data = [e.reset() for e in self.multi_envs]
        else:
            self.data = [e.reset(seed=seed+idx) for idx, e in enumerate(self.multi_envs)]

    def put(self, *args, **kwargs):
        for e in self.multi_envs:
            e.put(*args, **kwargs)

    def get(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.multi_envs]

    def close(self):
        for e in self.multi_envs:
            e.close()

def _unpack_shared_mem(shared_mem, n):
    np_buf = np.frombuffer(shared_mem.get_obj(), dtype=float)
    obs_arr = np_buf[:-3*n]
    rewards_arr = np_buf[-3*n:-2*n]
    terminals_arr = np_buf[-2*n:-n]
    truncated_arr = np_buf[-n:]

    return obs_arr, rewards_arr, terminals_arr, truncated_arr

def _worker_process(multi_env_cls, env_creator, env_args, env_kwargs,
        agents_per_env, envs_per_worker,
        worker_idx, shared_mem, send_pipe, recv_pipe):
    
    # I don't know if this helps. Sometimes it does, sometimes not.
    # Need to run more comprehensive tests
    #curr_process = psutil.Process()
    #curr_process.cpu_affinity([worker_idx])

    envs = multi_env_cls(env_creator, env_args, env_kwargs, n=envs_per_worker)
    obs_arr, rewards_arr, terminals_arr, truncated_arr = _unpack_shared_mem(
        shared_mem, agents_per_env * envs_per_worker)

    while True:
        request, args, kwargs = recv_pipe.recv()
        func = getattr(envs, request)
        response = func(*args, **kwargs)
        info = {}

        # TODO: Handle put/get
        if request in 'step reset'.split():
            obs, reward, done, truncated, info = response

            # TESTED: There is no overhead associated with 4 assignments to shared memory
            # vs. 4 assigns to an intermediate numpy array and then 1 assign to shared memory
            obs_arr[:] = obs.ravel()
            rewards_arr[:] = reward.ravel()
            terminals_arr[:] = done.ravel()
            truncated_arr[:] = truncated.ravel()

        send_pipe.send(info)


class Multiprocessing:
    '''Runs environments in parallel using multiprocessing

    Use this vectorization module for most applications
    '''
    reset = reset
    step = step
    single_observation_space = property(single_observation_space)
    single_action_space = property(single_action_space)
    structured_observation_space = property(structured_observation_space)
    flat_observation_space = property(flat_observation_space)
    unpack_batched_obs = unpack_batched_obs

    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            num_envs: int = 1,
            envs_per_worker: int = 1,
            envs_per_batch: int = None,
            env_pool: bool = False,
            mask_agents: bool = False,
            ) -> None:
        driver_env, multi_env_cls, agents_per_env = setup(
            env_creator, env_args, env_kwargs)
        num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker = calc_scale_params(
            num_envs, envs_per_batch, envs_per_worker, agents_per_env)

        agents_per_worker = agents_per_env * envs_per_worker
        observation_size = int(np.prod(_single_observation_space(driver_env).shape))
        observation_dtype = _single_observation_space(driver_env).dtype

        # Shared memory for obs, rewards, terminals, truncateds
        from multiprocessing import Process, Manager, Pipe, Array
        shared_mem = [
            Array('d', agents_per_worker*(3+observation_size))
            for _ in range(num_workers)
        ]
        main_send_pipes, work_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])
        work_send_pipes, main_recv_pipes = zip(*[Pipe() for _ in range(num_workers)])

        num_cores = psutil.cpu_count()
        processes = [Process(
            target=_worker_process,
            args=(multi_env_cls, env_creator, env_args, env_kwargs,
                  agents_per_env, envs_per_worker,
                  i%(num_cores-1), shared_mem[i],
                  work_send_pipes[i], work_recv_pipes[i])
            ) for i in range(num_workers)
        ]

        for p in processes:
            p.start()

        # Register all receive pipes with the selector
        import selectors
        sel = selectors.DefaultSelector()
        for pipe in main_recv_pipes:
            sel.register(pipe, selectors.EVENT_READ)

        self.processes = processes
        self.sel = sel
        self.observation_size = observation_size
        self.observation_dtype = observation_dtype
        self.shared_mem = shared_mem
        self.send_pipes = main_send_pipes
        self.recv_pipes = main_recv_pipes
        self.driver_env = driver_env
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.workers_per_batch = workers_per_batch
        self.envs_per_batch = envs_per_batch
        self.envs_per_worker = envs_per_worker
        self.agents_per_batch = agents_per_batch
        self.agents_per_worker = agents_per_worker
        self.agents_per_env = agents_per_env
        self.async_handles = None
        self.flag = RESET
        self.prev_env_id = []
        self.env_pool = env_pool
        self.mask_agents = mask_agents

    def recv(self):
        recv_precheck(self)
        recvs = []
        next_env_id = []
        if self.env_pool:
            while len(recvs) < self.workers_per_batch:
                for key, _ in self.sel.select(timeout=None):
                    response_pipe = key.fileobj
                    env_id = self.recv_pipes.index(response_pipe)

                    if response_pipe.poll():
                        info = response_pipe.recv()
                        o, r, d, t = _unpack_shared_mem(
                            self.shared_mem[env_id], self.agents_per_env * self.envs_per_worker)
                        o = o.reshape(
                            self.agents_per_env*self.envs_per_worker,
                            self.observation_size).astype(self.observation_dtype)

                        recvs.append((o, r, d, t, info, env_id))
                        next_env_id.append(env_id)

                    if len(recvs) == self.workers_per_batch:                    
                        break
        else:
            for env_id in range(self.workers_per_batch):
                response_pipe = self.recv_pipes[env_id]
                info = response_pipe.recv()
                o, r, d, t = _unpack_shared_mem(
                    self.shared_mem[env_id], self.agents_per_env * self.envs_per_worker)
                o = o.reshape(
                    self.agents_per_env*self.envs_per_worker,
                    self.observation_size).astype(self.observation_dtype)

                recvs.append((o, r, d, t, info, env_id))
                next_env_id.append(env_id)

        self.prev_env_id = next_env_id
        return aggregate_recvs(self, recvs)

    def send(self, actions):
        send_precheck(self)
        actions = split_actions(self, actions)
        for i, atns in zip(self.prev_env_id, actions):
            self.send_pipes[i].send(("step", [atns], {}))

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            for pipe in self.send_pipes:
                pipe.send(("reset", [], {}))
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
    structured_observation_space = property(structured_observation_space)
    flat_observation_space = property(flat_observation_space)
    unpack_batched_obs = unpack_batched_obs

    def __init__(self,
            env_creator: callable = None,
            env_args: list = [],
            env_kwargs: dict = {},
            num_envs: int = 1,
            envs_per_worker: int = 1,
            envs_per_batch: int = None,
            env_pool: bool = False,
            mask_agents: bool = False,
            ) -> None:
        driver_env, multi_env_cls, agents_per_env = setup(
            env_creator, env_args, env_kwargs)
        num_workers, workers_per_batch, envs_per_batch, agents_per_batch, agents_per_worker = calc_scale_params(
            num_envs, envs_per_batch, envs_per_worker, agents_per_env)

        import ray
        if not ray.is_initialized():
            import logging
            ray.init(
                include_dashboard=False,  # WSL Compatibility
                logging_level=logging.ERROR,
            )

        multi_envs = [
            ray.remote(multi_env_cls).remote(
                env_creator, env_args, env_kwargs, envs_per_worker
            ) for _ in range(num_workers)
        ]

        self.multi_envs = multi_envs
        self.driver_env = driver_env
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.workers_per_batch = workers_per_batch
        self.envs_per_batch = envs_per_batch
        self.envs_per_worker = envs_per_worker
        self.agents_per_batch = agents_per_batch
        self.agents_per_worker = agents_per_worker
        self.agents_per_env = agents_per_env
        self.async_handles = None
        self.flag = RESET
        self.ray = ray
        self.prev_env_id = []
        self.env_pool = env_pool
        self.mask_agents = mask_agents

    def recv(self):
        recv_precheck(self)
        recvs = []
        next_env_id = []
        if self.env_pool:
            recvs = self.ray.get(self.async_handles)
            env_id = [_ for _ in range(self.workers_per_batch)]
        else:
            ready, busy = self.ray.wait(
                self.async_handles, num_returns=self.workers_per_batch)
            env_id = [self.async_handles.index(e) for e in ready]
            recvs = self.ray.get(ready)

        recvs = [(o, r, d, t, i, eid)
            for (o, r, d, t, i), eid in zip(recvs, env_id)]
        self.prev_env_id = env_id
        return aggregate_recvs(self, recvs)

    def send(self, actions):
        send_precheck(self)
        actions = split_actions(self, actions)
        self.async_handles = [e.step.remote(a) for e, a in zip(self.multi_envs, actions)]

    def async_reset(self, seed=None):
        reset_precheck(self)
        if seed is None:
            self.async_handles = [e.reset.remote() for e in self.multi_envs]
        else:
            self.async_handles = [e.reset.remote(seed=seed+idx)
                for idx, e in enumerate(self.multi_envs)]

    def put(self, *args, **kwargs):
        for e in self.multi_envs:
            e.put.remote(*args, **kwargs)

    def get(self, *args, **kwargs):
        return self.ray.get([e.get.remote(*args, **kwargs) for e in self.multi_envs])

    def close(self):
        self.ray.get([e.close.remote() for e in self.multi_envs])
        self.ray.shutdown()
