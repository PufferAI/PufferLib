from pdb import set_trace as T

import numpy as np
import itertools
from multiprocessing import Process, Queue, shared_memory

RESET = 0
SEND = 1
RECV = 2

class RemoteEnvs:
    def __init__(self, request_queue, response_queue, env_creator, n, obs_np):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.envs = [env_creator() for _ in range(n)]
        self.obs_np = obs_np

    def process_requests(self):
        while True:
            request, args, kwargs = self.request_queue.get()
            if request == "seed":
                self.seed(*args, **kwargs)
            elif request == "profile_all":
                self.response_queue.put(self.profile_all(*args, **kwargs))
            elif request == "put_all":
                self.put_all(*args, **kwargs)
            elif request == "get_all":
                self.response_queue.put(self.get_all(*args, **kwargs))
            elif request == "reset_all":
                self.response_queue.put(self.reset_all(*args, **kwargs))
            elif request == "step":
                self.response_queue.put(self.step(*args, **kwargs))
            elif request == "terminate":
                break

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)
            seed += 1

    def profile_all(self):
        return [e.timers for e in self.envs]

    def put_all(self, *args, **kwargs):
        for e in self.envs:
            e.put(*args, **kwargs)

    def get_all(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.envs]

    def reset_all(self, seed=None):
        async_handles = []
        env_index = 0
        for e in self.envs:
            obs = e.reset(seed=seed)
            if seed is not None:
                seed += 1

            obs_list = []
            for k, v in obs.items():
                obs_list.append(v)

            stacked_obs = np.stack(obs_list)
            self.obs_np[env_index] = stacked_obs

            rewards_dict = {k: 0 for k in obs}
            dones_dict = {k: False for k in obs}
            infos_list = [{} for _ in obs]

            async_handles.append((rewards_dict, dones_dict, infos_list))
            env_index += 1

        return async_handles

    def step(self, actions_lists):
        returns = []
        assert len(self.envs) == len(actions_lists)
        env_index = 0
        for env, actions in zip(self.envs, actions_lists):
            if env.done:
                obs = env.reset()
                rewards = {k: 0 for k in obs}
                dones = {k: False for k in obs}
                infos = {}
            else:
                obs, rewards, dones, infos = env.step(actions)

            stacked_obs = np.stack(list(obs.values()))
            self.obs_np[env_index] = stacked_obs

            returns.append((rewards, dones, infos))
            env_index += 1

        return returns

def make_remote_envs(env_creator, n, obs_np):
    request_queue, response_queue = Queue(), Queue()
    remote_env = RemoteEnvs(request_queue, response_queue, env_creator, n, obs_np)
    process = Process(target=remote_env.process_requests)
    process.start()
    return request_queue, response_queue, process


class VecEnv:
    def __init__(self, binding, num_workers, envs_per_worker=1):
        '''Creates env_per_worker serial environments on each of num_workers remote processes

        Synchronous API: Use env.reset() and env.step(actions)

        Asynchronous API: Use env.async_reset(), env.send(actions), and env.recv(actions)
        This confers no advantage unless you are using multiple VecEnvs in a double-buffered
        or multi-buffered configuration. See the PufferLib custom CleanRL demo for an example.

        Args:
            binding: A pufferlib.emulation.Binding object
            num_workers: The number of remote processes to create
            envs_per_worker: The number of serial environments to create on each remote process
        '''
        assert envs_per_worker > 0, 'Each worker must have at least 1 env'
        assert type(envs_per_worker) == int

        self.binding = binding
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.state = RESET

        self.total_envs = num_workers * envs_per_worker
        single_obs_shape = self.single_observation_space.shape
        self.obs_shape = (self.total_envs, *single_obs_shape)
        self.obs_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.obs_shape) * np.dtype(np.float32).itemsize)
        self.obs_np = np.ndarray(self.obs_shape, dtype=np.float32, buffer=self.obs_shm.buf)

        self.remote_envs_lists = [
            make_remote_envs(
                self.binding.env_creator,
                self.envs_per_worker,
                self.obs_np[idx * self.envs_per_worker:(idx + 1) * self.envs_per_worker],
            ) for idx in range(self.num_workers)
        ]

    @property
    def single_observation_space(self):
        '''Convenience function; returns self.binding.single_observation_space'''
        return self.binding.single_observation_space

    @property
    def single_action_space(self):
        '''Convenience function; returns self.binding.single_action_space'''
        return self.binding.single_action_space

    def close(self):
        for request_queue, _, _ in self.remote_envs_lists:
            request_queue.put(("terminate", None, None))

        self.obs_shm.close()
        self.obs_shm.unlink()

        for _, _, process in self.remote_envs_lists:
            process.join()

    def profile(self):
        '''Returns profiling timers from all remote environments'''
        return list(itertools.chain.from_iterable([e.profile_all.remote() for e in self.remote_envs_lists]))


    def async_reset(self, seed=None):
        '''Asynchronously reset environments. Does not block.'''
        assert self.state == RESET, 'Call reset only once on initialization'
        self.state = RECV

        self.async_handles = []

        for request_queue, _, _ in self.remote_envs_lists:
            request_queue.put(("reset_all", (seed,), {}))
            if seed is not None:
                seed += self.envs_per_worker * self.binding.max_agents

    def recv(self):
        assert self.state == RECV, 'Call reset before stepping'
        self.state = SEND

        self.agent_keys = []
        rewards, dones, infos = {}, {}, []
        for envs_lists in self.remote_envs_lists:
            _, response_queue, _ = envs_lists
            async_handle = response_queue.get()  # This line is blocking

            a_keys = []
            for r, d, i in async_handle:
                a_keys.append(list(r.keys()))
                rewards.update(r)
                dones.update(d)
                infos.append(i)

            self.agent_keys.append(a_keys)

        obs = self.obs_np.copy()

        return obs, rewards, dones, infos

    def send(self, actions, env_id=None):
        '''Send observations to remote async environments. Does not block.'''
        assert self.state == SEND, 'Call reset + recv before send'
        self.state = RECV

        if type(actions) == list:
            actions = np.array(actions)

        actions = np.array_split(actions, self.num_workers)

        for i, envs_lists in enumerate(self.remote_envs_lists):
            request_queue, response_queue, _ = envs_lists
            atns_list = np.split(actions[i], self.envs_per_worker)
            keys_list = self.agent_keys[i]
            atns_list = [dict(zip(keys, atns)) for keys, atns in zip(keys_list, atns_list)]
            request_queue.put(("step", (atns_list,), {}))

    def reset(self, seed=None):
        '''Syncronously resets remote environments. Blocks.'''
        self.async_reset()
        return self.recv()[0]

    def step(self, actions):
        '''Syncronously steps remote environments. Blocks.'''
        self.send(actions)
        return self.recv()