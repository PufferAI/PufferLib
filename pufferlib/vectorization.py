from pdb import set_trace as T

import numpy as np
import itertools

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


class MultiEnv:
    '''Runs multiple environments in serial'''
    def __init__(self, env_creator, n):
        self.envs = [env_creator() for _ in range(n)]
        self.agent_keys = None
        self.preallocated_obs = None

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

        return self.preallocated_obs

    def step(self, actions):
        assert len(self.envs) == len(actions)
        actions = np.array_split(actions, len(self.envs))
        obs, rewards, dones, infos = [], [], [], []
        
        for idx, (a_keys, env, atns) in enumerate(zip(self.agent_keys, self.envs, actions)):
            if env.done:
                obs = env.reset()
                rewards = [0 for _ in obs]
                dones = [False for _ in obs]
                infos = {}
            else:
                atns = dict(zip(a_keys, atns))
                o, r, d, i= env.step(atns)

            self.agent_keys[idx] = list(o.keys())

            for ii, oo in enumerate(o.values()):
                self.preallocated_obs[ii] = oo

            rewards.extend(r.values())
            dones.extend(d.values())
            infos.append(i)

        return self.preallocated_obs, rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()


class VecEnv:
    def __init__(self, binding, backend_cls, num_workers, envs_per_worker=1):
        assert envs_per_worker > 0, 'Each worker must have at least 1 env'
        assert type(envs_per_worker) == int

        self.binding = binding
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker

        self.state = RESET

        self.backends = [
            backend_cls(self.binding.env_creator, envs_per_worker)
            for _ in range(self.num_workers)
        ]

    @property
    def single_observation_space(self):
        return self.binding.single_observation_space

    @property
    def single_action_space(self):
        return self.binding.single_action_space

    def close(self):
        for backend in self.backends:
            backend.close()

    def profile(self):
        return list(itertools.chain.from_iterable([e.profile_all() for e in self.backends]))

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

        backends_recv = [backend.recv() for backend in self.backends]
        
        obs = np.array([o.values() for o, _, _, _ in backends_recv])
        rewards = [r.values() for _, r, _, _ in backends_recv]
        dones = [d.values() for _, _, d, _ in backends_recv]
        infos = [i for _, _, _, i in backends_recv]

        return obs, rewards, dones, infos

    def send(self, actions, env_id=None):
        assert self.state == SEND, 'Call reset + recv before send'
        self.state = RECV

        if type(actions) == list:
            actions = np.array(actions)

        actions_split = np.array_split(actions, self.num_workers)

        for backend, keys_list, atns_list in zip(self.backends, self.agent_keys, actions_split):
            atns_list_split = np.array_split(atns_list, self.envs_per_worker)
            atns_list_dicts = [dict(zip(keys, atns)) for keys, atns in zip(keys_list, atns_list_split)]
            backend.send(atns_list_dicts)

    def reset(self, seed=None):
        self.async_reset()
        return self.recv()[0]

    def step(self, actions):
        self.send(actions)
        return self.recv()


class Serial(MultiEnv, Backend):
    def __init__(self, env_creator, n):
        super().__init__(env_creator, n)
        self.async_handles = None

    def async_reset(self, seed=None):
        assert self.async_handles is None, 'reset called after send'
        self.async_handles = super().reset(seed=seed)

    def send(self, actions_lists):
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

    def profile_all(self):
        self.request_queue.put(("profile_all", [], {}))
        return self.response_queue.get()

    def put_all(self, *args, **kwargs):
        self.request_queue.put(("put_all", args, kwargs))

    def get_all(self, *args, **kwargs):
        self.request_queue.put(("get_all", args, kwargs))
        return self.response_queue.get()

    def close(self):
        self.request_queue.put(("close", [], {}))

    def async_reset_all(self, seed=None):
        self.request_queue.put(("reset_all", [seed], {}))

    def reset_all(self, seed=None):
        self.request_queue.put(("reset_all", [seed], {}))
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
        self.remote_env = ray.remote(MultiEnv).remote(env_creator, n)
        self.ray = ray

    def seed(self, seed):
        return self.ray.get(self.remote_env.seed.remote(seed))

    def profile_all(self):
        return self.ray.get(self.remote_env.profile_all.remote())

    def put_all(self, *args, **kwargs):
        return self.ray.get(self.remote_env.put_all.remote(*args, **kwargs))

    def get_all(self, *args, **kwargs):
        return self.ray.get(self.remote_env.get_all.remote(*args, **kwargs))

    def close(self):
        return self.ray.get(self.remote_env.close.remote())

    def async_reset_all(self, seed=None):
        self.future = self.remote_env.reset_all.remote(seed)

    def reset_all(self, seed=None):
        return self.ray.get(self.remote_env.reset_all.remote(seed))

    def step(self, actions_lists):
        return self.ray.get(self.remote_env.step.remote(actions_lists))

    def send(self, actions_lists):
        self.future = self.remote_env.step.remote(actions_lists)

    def recv(self):
        return self.ray.get(self.future)