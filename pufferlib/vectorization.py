from pdb import set_trace as T

import numpy as np
import itertools
import inspect

import pufferlib.emulation
from pufferlib.emulation import GymPufferEnv, PettingZooPufferEnv


RESET = 0
SEND = 1
RECV = 2

class Backend:
    def __init__(self, env_creator, n):
        raise NotImplementedError

    def send(self, actions):
        raise NotImplementedError
    
    def recv(self):
        raise NotImplementedError
    
    def async_reset(self, seed=None):
        raise NotImplementedError

    def profile_all(self):
        raise NotImplementedError

    def put(self, *args, **kwargs):
        raise NotImplementedError
    
    def get(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class SerialPufferEnvs:
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

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)
            seed += 1

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


class SerialGymPufferEnvs(SerialPufferEnvs):
    '''Runs multiple Puffer wrapped Gym environments in serial'''
    def reset(self, seed=None):
        for i, e in enumerate(self.envs):
            ob = e.reset(seed=seed)

            if self.preallocated_obs is None:
                self.preallocated_obs = np.empty((len(self.envs), *ob.shape), dtype=ob.dtype)

            self.preallocated_obs[i] = ob

            if seed is not None:
                seed += 1

        rewards = [0] * len(self.preallocated_obs)
        dones = [False] * len(self.preallocated_obs)
        infos = {}
 
        return self.preallocated_obs, rewards, dones, infos

    def step(self, actions):
        rewards, dones, infos = [], [], []
        
        for idx, (env, atns) in enumerate(zip(self.envs, actions)):
            if env.done:
                o  = env.reset()
                rewards.extend([0] * len(self.preallocated_obs))
                dones.extend([False] * len(self.preallocated_obs))
                infos.append({})
            else:
                o, r, d, i = env.step(atns)
                rewards.append(r)
                dones.append(d)
                infos.append(i)

            self.preallocated_obs[idx] = o

        return self.preallocated_obs, rewards, dones, infos


class SerialPettingZooPufferEnvs(SerialPufferEnvs):
    '''Runs multiple Puffer wrapped Petting Zoo envs in serial'''
    def __init__(self, env_creator, env_args=[], env_kwargs={}, n=1):
        super().__init__(env_creator, env_args, env_kwargs, n)
        self.agent_keys = None

    @property
    def single_observation_space(self):
        return self.envs[0].observation_space(self.envs[0].possible_agents[0])

    @property
    def single_action_space(self):
        return self.envs[0].action_space(self.envs[0].possible_agents[0])
  
    def reset(self, seed=None):
        self.agent_keys = []
        for e in self.envs:
            obs = e.reset(seed=seed)
            self.agent_keys.append(list(obs.keys()))

            if self.preallocated_obs is None:
                ob = obs[list(obs.keys())[0]]
                self.preallocated_obs = np.empty((len(self.envs)*len(obs), *ob.shape), dtype=ob.dtype)

            for i, o in enumerate(obs.values()):
                self.preallocated_obs[i] = o

            if seed is not None:
                seed += 1

        rewards = [0] * len(self.preallocated_obs)
        dones = [False] * len(self.preallocated_obs)
        infos = {}
 
        return self.preallocated_obs, rewards, dones, infos

    def step(self, actions):
        actions = np.array_split(actions, len(self.envs))
        rewards, dones, infos = [], [], []
        
        for idx, (a_keys, env, atns) in enumerate(zip(self.agent_keys, self.envs, actions)):
            if env.done:
                o  = env.reset()
                rewards.extend([0] * len(self.preallocated_obs))
                dones.extend([False] * len(self.preallocated_obs))
                infos.append({})
            else:
                assert len(a_keys) == len(atns)
                atns = dict(zip(a_keys, atns))
                o, r, d, i= env.step(atns)
                rewards.extend(r.values())
                dones.extend(d.values())
                infos.append(i)

            self.agent_keys[idx] = list(o.keys())

            for ii, oo in enumerate(o.values()):
                self.preallocated_obs[ii] = oo

        return self.preallocated_obs, rewards, dones, infos


class VecEnv:
    def __init__(self, env_creator, env_args=[], env_kwargs={},
            num_workers=1, envs_per_worker=1):
        # Figure out whether to use Gym or PettingZoo as a backend
        if inspect.isclass(env_creator):
            if issubclass(env_creator, GymPufferEnv):
                self._backend = SerialGymPufferEnvs
            elif issubclass(env_creator, PettingZooPufferEnv):
                self._backend = SerialPettingZooPufferEnvs
            else:
                raise TypeError('env_creator must be a GymPufferEnv or PettingZooPufferEnv class')
        elif callable(env_creator):
            created_env = env_creator(*env_args, **env_kwargs)
            if isinstance(created_env, GymPufferEnv):
                self._backend = SerialGymPufferEnvs
            elif isinstance(created_env, PettingZooPufferEnv):
                self._backend = SerialPettingZooPufferEnvs
            else:
                raise TypeError('created env must be an instance of GymPufferEnv or PettingZooPufferEnv')
        else:
            raise TypeError('env_creator must be a callable or a class')
       
        assert envs_per_worker > 0, 'Each worker must have at least 1 env'
        assert type(envs_per_worker) == int

        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker

        self.state = RESET
        self.preallocated_obs = None
        self.num_agents = None

        self.backends = [
            puffer_env_cls(envs_per_worker)
            for _ in range(self.num_workers)
        ]

    def close(self):
        for backend in self.backends:
            backend.close()

    def profile(self):
        return list(itertools.chain.from_iterable([e.profile() for e in self.backends]))

    def async_reset(self, seed=None):
        assert self.state == RESET, 'Call reset only once on initialization'
        self.state = RECV

        for backend in self.backends:
            backend.async_reset(seed=seed)
            if seed is not None:
                seed += self.envs_per_worker * self.binding.max_agents

    def recv(self):
        assert self.state == RECV, 'Call reset before stepping'
        self.state = SEND

        returns = [backend.recv() for backend in self.backends]
        obs, rewards, dones, infos = list(zip(*returns))

        if self.preallocated_obs is None:
            ob = obs[0]
            self.num_agents = len(ob)
            self.preallocated_obs = np.empty((len(obs)*self.num_agents, *ob.shape[1:]), dtype=ob.dtype)

        for i, o in enumerate(obs):
            self.preallocated_obs[i*self.num_agents:(i+1)*self.num_agents] = o

        rewards = list(itertools.chain.from_iterable(rewards))
        dones = list(itertools.chain.from_iterable(dones))
        infos = list(itertools.chain.from_iterable(infos))

        return self.preallocated_obs, rewards, dones, infos

    def send(self, actions, env_id=None):
        assert self.state == SEND, 'Call reset + recv before send'
        self.state = RECV

        if type(actions) == list:
            actions = np.array(actions)

        actions_split = np.array_split(actions, self.num_workers)

        for backend, atns in zip(self.backends, actions_split):
            backend.send(atns)

    def reset(self, seed=None):
        self.async_reset()
        return self.recv()[0]

    def step(self, actions):
        self.send(actions)
        return self.recv()

 
class Serial(SerialGymPufferEnvs, Backend):
    def __init__(self, env_creator=None, env_args=[], env_kwargs={}, n=1):
        super().__init__(env_creator, env_args, env_kwargs, n)
        self.async_handles = None

    def async_reset(self, seed=None):
        assert self.async_handles is None, 'reset called after send'
        self.async_handles = super().reset(seed=seed)

    def send(self, actions_lists, env_id=None):
        assert self.async_handles is None, 'send called before recv'
        self.async_handles = super().step(actions_lists)

    def recv(self):
        assert self.async_handles is not None, 'recv called before reset or send'
        async_handles = self.async_handles
        self.async_handles = None
        return async_handles


class Multiprocessing(Backend):
    def __init__(self, env_creator, n):
        from multiprocessing import Process, Queue
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.process = Process(target=self._worker_process, args=(env_creator, n, self.request_queue, self.response_queue))
        self.process.start()

    def _worker_process(self, env_creator, n, request_queue, response_queue):
        self.envs = MultiEnv(env_creator, n)

        while True:
            request, args, kwargs = request_queue.get()
            func = getattr(self.envs, request)
            response = func(*args, **kwargs)
            response_queue.put(response)

    def seed(self, seed):
        self.request_queue.put(("seed", [seed], {}))

    def profile(self):
        self.request_queue.put(("profile", [], {}))
        return self.response_queue.get()

    def put(self, *args, **kwargs):
        self.request_queue.put(("put", args, kwargs))

    def get(self, *args, **kwargs):
        self.request_queue.put(("get", args, kwargs))
        return self.response_queue.get()

    def close(self):
        self.request_queue.put(("close", [], {}))

    def async_reset(self, seed=None):
        self.request_queue.put(("reset", [seed], {}))

    def reset(self, seed=None):
        self.request_queue.put(("reset", [seed], {}))
        return self.response_queue.get()

    def step(self, actions_lists):
        self.send(actions_lists)
        return self.recv()

    def send(self, actions_lists):
        self.request_queue.put(("step", [actions_lists], {}))

    def recv(self):
        return self.response_queue.get()


class SharedMemoryMultiprocessing(Multiprocessing):
    def __init__(self, env_creator, n, obs_shape):
        from multiprocessing import shared_memory
        super().__init__(env_creator, n)
        self.obs_shape = (n,) + obs_shape
        self.obs_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.obs_shape) * np.dtype(np.float32).itemsize)
        self.obs_np = np.ndarray(self.obs_shape, dtype=np.float32, buffer=self.obs_shm.buf)

    def _worker_process(self, env_creator, n, request_queue, response_queue):
        self.envs = MultiEnv(env_creator, n)

        while True:
            request, args, kwargs = request_queue.get()

            if request == "terminate":
                self.envs.close()
                self.obs_shm.close()
                break

            elif request == "step":
                actions_lists = args[0]
                results = self.envs.step(actions_lists)

                for i, (obs, _, _, _) in enumerate(results):
                    self.obs_np[i] = obs

                dones = [result[2] for result in results]
                response_queue.put((len(results), dones))

            else:
                func = getattr(self.envs, request)
                response = func(*args, **kwargs)
                response_queue.put(response)

    def step(self, actions_lists):
        self.send(actions_lists)
        return self.recv()

    def recv(self):
        num_new_obs, dones = self.response_queue.get()
        return self.obs_np[:num_new_obs], dones


class Ray(Backend):
    def __init__(self, env_creator, n):
        import ray
        ray.init(
            include_dashboard=False,  # WSL Compatibility
            ignore_reinit_error=True,
        )
        @ray.remote
        class RemoteMultiEnv(MultiEnv):
            pass

        self.remote_env = RemoteMultiEnv.remote(env_creator, n)
        #self.remote_env = ray.remote(MultiEnv).remote(env_creator, n)
        self.ray = ray

    def seed(self, seed):
        return self.ray.get(self.remote_env.seed.remote(seed))

    def profile(self):
        return self.ray.get(self.remote_env.profile.remote())

    def put(self, *args, **kwargs):
        return self.ray.get(self.remote_env.put.remote(*args, **kwargs))

    def get(self, *args, **kwargs):
        return self.ray.get(self.remote_env.get.remote(*args, **kwargs))

    def close(self):
        return self.ray.get(self.remote_env.close.remote())

    def async_reset(self, seed=None):
        self.future = self.remote_env.reset.remote(seed)

    def reset_all(self, seed=None):
        return self.ray.get(self.remote_env.reset.remote(seed))

    def step(self, actions_lists):
        return self.ray.get(self.remote_env.step.remote(actions_lists))

    def send(self, actions_lists):
        self.future = self.remote_env.step.remote(actions_lists)

    def recv(self):
        return self.ray.get(self.future)