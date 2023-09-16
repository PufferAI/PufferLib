from pdb import set_trace as T

import numpy as np
import itertools
import inspect
from abc import ABC

import pufferlib.emulation
from pufferlib.emulation import GymPufferEnv, PettingZooPufferEnv


RESET = 0
SEND = 1
RECV = 2


class MultiEnv(ABC):
    '''Abstract base class for running multiple Puffer wrapped environments in serial'''
    def __init__(self, env_creator, env_args=[], env_kwargs={}, n=1):
        self.envs = [env_creator(*env_args, **env_kwargs) for _ in range(n)]
        self.preallocated_obs = None

    @property
    def single_observation_space(self):
        return self.envs[0].observation_space
 
    @property
    def single_action_space(self):
        return self.envs[0].action_space

    @property
    def structured_observation_space(self):
        return self.envs[0].structured_observation_space

    @property
    def flat_observation_space(self):
        return self.envs[0].flat_observation_space

    def profile(self):
        return [e.timers for e in self.envs]

    def put(self, *args, **kwargs):
        for e in self.envs:
            e.put(*args, **kwargs)
        
    def get(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.envs]

    def close(self):
        for env in self.envs:
            env.close()


class GymMultiEnv(MultiEnv):
    '''Runs multiple Puffer wrapped Gym environments in serial'''
    def reset(self, seed=None):
        for idx, e in enumerate(self.envs):
            if seed is None:
                ob = e.reset()
            else:
                ob = e.reset(seed=hash(1000*seed + idx))

            if self.preallocated_obs is None:
                self.preallocated_obs = np.empty((len(self.envs), *ob.shape), dtype=ob.dtype)

            self.preallocated_obs[idx] = ob

        rewards = [0] * len(self.preallocated_obs)
        dones = [False] * len(self.preallocated_obs)
        infos = [{} for _ in self.envs]
 
        return self.preallocated_obs, rewards, dones, infos

    def step(self, actions):
        rewards, dones, infos = [], [], []
        
        for idx, (env, atns) in enumerate(zip(self.envs, actions)):
            if env.done:
                o  = env.reset()
                rewards.append(0)
                dones.append(False)
                infos.append({})
            else:
                o, r, d, i = env.step(atns)
                rewards.append(r)
                dones.append(d)
                infos.append(i)

            self.preallocated_obs[idx] = o

        return self.preallocated_obs, rewards, dones, infos


class PettingZooMultiEnv(MultiEnv):
    '''Runs multiple Puffer wrapped Petting Zoo envs in serial'''
    def __init__(self, env_creator, env_args=[], env_kwargs={}, n=1):
        super().__init__(env_creator, env_args, env_kwargs, n)
        self.agent_keys = None
 
    def reset(self, seed=None):
        self.agent_keys = []

        ptr = 0
        for idx, e in enumerate(self.envs):
            if seed is None:
                obs = e.reset()
            else:
                obs = e.reset(seed=hash(1000*seed + idx))

            self.agent_keys.append(list(obs.keys()))

            if self.preallocated_obs is None:
                ob = obs[list(obs.keys())[0]]
                self.preallocated_obs = np.empty((len(self.envs)*len(obs), *ob.shape), dtype=ob.dtype)

            for o in obs.values():
                self.preallocated_obs[ptr] = o
                ptr += 1

        rewards = [0] * len(self.preallocated_obs)
        dones = [False] * len(self.preallocated_obs)
        infos = [
            {agent_id: {} for agent_id in self.envs[0].possible_agents}
            for _ in self.envs
        ]
 
        return self.preallocated_obs, rewards, dones, infos

    def step(self, actions):
        actions = np.array_split(actions, len(self.envs))
        rewards, dones, infos = [], [], []

        ptr = 0        
        for idx, (a_keys, env, atns) in enumerate(zip(self.agent_keys, self.envs, actions)):
            if env.done:
                o  = env.reset()
                num_agents = len(env.possible_agents)
                rewards.extend([0] * num_agents)
                dones.extend([False] * num_agents)
                infos.append({agent_id: {} for agent_id in env.possible_agents})
            else:
                assert len(a_keys) == len(atns)
                atns = dict(zip(a_keys, atns))
                o, r, d, i= env.step(atns)
                rewards.extend(r.values())
                dones.extend(d.values())
                infos.append(i)

            self.agent_keys[idx] = list(o.keys())

            for oo in o.values():
                self.preallocated_obs[ptr] = oo
                ptr += 1

        return self.preallocated_obs, rewards, dones, infos


class VecEnv(ABC):
    '''Abstract base class for the vectorization API
    
    Contains shared code for splitting/aggregating data across multiple environments
    '''
    def __init__(self,
            env_creator=None, env_args=[], env_kwargs={},
            num_workers=1, envs_per_worker=1):

        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker

        self.async_handles = None
        self.state = RESET

        # Determine if the env uses Gym or PettingZoo
        self.driver_env = env_creator(*env_args, **env_kwargs)
        if isinstance(self.driver_env, GymPufferEnv):
            self._wrapper = GymMultiEnv
            self._num_agents = 1
        elif isinstance(self.driver_env, PettingZooPufferEnv):
            self._wrapper = PettingZooMultiEnv
            self._num_agents = len(self.driver_env.possible_agents)
        else:
            raise TypeError('env_creator must return an instance of GymPufferEnv or PettingZooPufferEnv')

        # Preallocate storeage for observations
        self.preallocated_obs = np.empty((
            self.num_agents * num_workers * envs_per_worker,
            *self.single_observation_space.shape,
        ), dtype=self.single_observation_space.dtype)

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def single_observation_space(self):
        if self._wrapper == GymMultiEnv:
            return self.driver_env.observation_space
        return self.driver_env.single_observation_space
 
    @property
    def single_action_space(self):
        if self._wrapper == GymMultiEnv:
            return self.driver_env.action_space
        return self.driver_env.single_action_space

    @property
    def structured_observation_space(self):
        return self.driver_env.structured_observation_space

    @property
    def flat_observation_space(self):
        return self.driver_env.flat_observation_space

    def unpack_batched_obs(self, obs):
        return self.driver_env.unpack_batched_obs(obs)

    def send(self, actions, env_id=None):
        assert self.state == SEND, 'Call reset + recv before send'
        self.state = RECV

        if type(actions) == list:
            actions = np.array(actions)

        assert isinstance(actions, np.ndarray), 'Actions must be a numpy array'

        assert len(actions) == self.num_agents * self.num_workers * self.envs_per_worker
        actions_split = np.array_split(actions, self.num_workers)
        self._send(actions_split)
  
    def recv(self):
        assert self.state == RECV, 'Call reset before stepping'
        self.state = SEND

        returns = self._recv()
        obs, rewards, dones, infos = list(zip(*returns))

        for i, o in enumerate(obs):
            total_agents = self.num_agents * self.envs_per_worker
            self.preallocated_obs[i*total_agents:(i+1)*total_agents] = o

        rewards = list(itertools.chain.from_iterable(rewards))
        dones = list(itertools.chain.from_iterable(dones))
        orig = infos
        infos = [i for ii in infos for i in ii]

        assert len(rewards) == len(dones) == total_agents * self.num_workers
        assert len(infos) == self.num_workers * self.envs_per_worker
        return self.preallocated_obs, rewards, dones, infos

    def async_reset(self, seed=None):
        assert self.state == RESET, 'Call reset only once on initialization'
        self.state = RECV

        self._async_reset(seed=seed)

    def profile(self):
        return list(itertools.chain.from_iterable([self._profile()]))

    def reset(self, seed=None):
        self.async_reset()
        return self.recv()[0]

    def step(self, actions):
        self.send(actions)
        return self.recv()

    def put(self, *args, **kwargs):
        raise NotImplementedError
    
    def get(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError



class Serial(VecEnv):
    def __init__(self,
            env_creator=None, env_args=[], env_kwargs={},
            num_workers=1, envs_per_worker=1):
        '''Runs environments in serial on the main process
        
        Use this vectorization module for debugging environments
        '''
        super().__init__(env_creator, env_args, env_kwargs,
            num_workers, envs_per_worker)

        self.envs = [
            self._wrapper(env_creator, env_args, env_kwargs, envs_per_worker)
            for _ in range(num_workers)
        ]

    def _send(self, actions):
        self.data = [e.step(a) for e, a in zip(self.envs, actions)]
    
    def _recv(self):
        return self.data
    
    def _async_reset(self, seed=None):
        if seed is None:
            self.data = [e.reset() for e in self.envs]
        else:
            self.data = [e.reset(seed=seed+idx) for idx, e in enumerate(self.envs)]

    def _profile(self):
        return [e.profile() for e in self.envs]

    def put(self, *args, **kwargs):
        for e in self.envs:
            e.put(*args, **kwargs)
    
    def get(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.envs]

    def close(self):
        for e in self.envs:
            e.close()


class Multiprocessing(VecEnv):
    '''Runs environments in parallel on multiple processes
    
    Use this module for most applications
    '''
    def __init__(self,
            env_creator=None, env_args=[], env_kwargs={},
            num_workers=1, envs_per_worker=1):
        super().__init__(env_creator, env_args, env_kwargs,
            num_workers, envs_per_worker)

        from multiprocessing import Process, Queue
        self.request_queues = [Queue() for _ in range(num_workers)]
        self.response_queues = [Queue() for _ in range(num_workers)]

        self.processes = [Process(
            target=self._worker_process,
            args=(self._wrapper, env_creator, env_args, env_kwargs,
                  envs_per_worker, self.request_queues[i], self.response_queues[i])) 
            for i in range(num_workers)]

        for p in self.processes:
            p.start()

    def _worker_process(self, wrapper, env_creator, env_args, env_kwargs, n, request_queue, response_queue):
        envs = wrapper(env_creator, env_args, env_kwargs, n=n)

        while True:
            request, args, kwargs = request_queue.get()
            func = getattr(envs, request)
            response = func(*args, **kwargs)
            response_queue.put(response)

    def _send(self, actions_lists):
        for queue, actions in zip(self.request_queues, actions_lists):
            queue.put(("step", [actions], {}))

    def _recv(self):
        return [queue.get() for queue in self.response_queues]

    def _async_reset(self, seed=None):
        if seed is None:
            for queue in self.request_queues:
                queue.put(("reset", [], {}))
        else:
            for idx, queue in enumerate(self.request_queues):
                queue.put(("reset", [], {"seed": seed+idx}))

    def _profile(self):
        for queue in self.request_queues:
            queue.put(("profile", [], {}))

        return [queue.get() for queue in self.response_queues]

    def put(self, *args, **kwargs):
        for queue in self.request_queues:
            queue.put(("put", args, kwargs))

    def get(self, *args, **kwargs):
        for queue in self.request_queues:
            queue.put(("get", args, kwargs))
        
        return [queue.get() for queue in self.response_queues]

    def close(self):
        for queue in self.request_queues:
            queue.put(("close", [], {}))

        for p in self.processes:
            p.terminate()

        for p in self.processes:
            p.join()


class Ray(VecEnv):
    '''Runs environments in parallel on multiple processes using Ray

    Use this module for distributed simulation on a cluster. It can also be
    faster than multiprocessing on a single machine for specific environments.
    '''
    def __init__(self,
            env_creator=None, env_args=[], env_kwargs={},
            num_workers=1, envs_per_worker=1):
        super().__init__(env_creator, env_args, env_kwargs,
            num_workers, envs_per_worker)

        import ray
        self.ray = ray
        if not ray.is_initialized():
            import logging
            ray.init(
                include_dashboard=False,  # WSL Compatibility
                logging_level=logging.ERROR,
            )

        self.envs = [
            ray.remote(self._wrapper).remote(
                env_creator, env_args, env_kwargs, envs_per_worker
            ) for _ in range(num_workers)
        ]

    def _send(self, actions):
        self.async_handles = [e.step.remote(a) for e, a in zip(self.envs, actions)]
    
    def _recv(self):
        return self.ray.get(self.async_handles)
    
    def _async_reset(self, seed=None):
        if seed is None:
            self.async_handles = [e.reset.remote() for e in self.envs]
        else:
            self.async_handles = [e.reset.remote(seed=seed+idx) for idx, e in enumerate(self.envs)]

    def _profile(self):
        return self.ray.get([e.profile.remote() for e in self.envs])

    def put(self, *args, **kwargs):
        for e in self.envs:
            e.put.remote(*args, **kwargs)
    
    def get(self, *args, **kwargs):
        return self.ray.get([e.get.remote(*args, **kwargs) for e in self.envs])

    def close(self):
        for e in self.envs:
            e.close.remote()
