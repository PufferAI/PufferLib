from pdb import set_trace as T

import numpy as np
import itertools
from multiprocessing import Process, Pipe, Queue

RESET = 0
SEND = 1
RECV = 2

class RemoteEnvs:
    def __init__(self, request_queue, response_queue, env_creator, n):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.envs = [env_creator() for _ in range(n)]

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
            else:
                call_candidate = getattr(self.envs[0], request, None)
                if call_candidate is not None and callable(call_candidate):
                    self.response_queue.put(call_candidate(*args, **kwargs))
                else:
                    self.response_queue.put(None)

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)
            seed += 1
    
    def call(self, name, *args, **kwargs):
        return [getattr(e, name)(*args, **kwargs) for e in self.envs]

    def profile_all(self, *args, **kwargs):
        return [e.timers for e in self.envs]

    def put_all(self, *args, **kwargs):
        for e in self.envs:
            e.put(*args, **kwargs)

    def get_all(self, *args, **kwargs):
        return [e.get(*args, **kwargs) for e in self.envs]

    def reset_all(self, seed=None):
        async_handles = []
        for e in self.envs:
            async_handles.append((e.reset(seed=seed), {}, {}, {}))
            if seed is not None:
                seed += 1
        return async_handles

    def step(self, actions_lists):
        returns = []
        assert len(self.envs) == len(actions_lists)
        for env, actions in zip(self.envs, actions_lists):
            if env.done:
                obs = env.reset()
                rewards = {k: 0 for k in obs}
                dones = {k: False for k in obs}
                infos = {}
            else:
                obs, rewards, dones, infos = env.step(actions)

            returns.append((obs, rewards, dones, infos))

        return returns

def make_remote_envs(env_creator, n):
    request_queue, response_queue = Queue(), Queue()
    remote_env = RemoteEnvs(request_queue, response_queue, env_creator, n)
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

        self.remote_envs_lists = [
            make_remote_envs(
                self.binding.env_creator,
                self.envs_per_worker,
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
            request_queue.put(("terminate", [], {}))

        for _, _, process in self.remote_envs_lists:
            process.join()

    def profile(self):
        '''Returns profiling timers from all remote environments'''
        for request_queue, _, _ in self.remote_envs_lists:
            request_queue.put(("profile_all", [], {}))

        profile_results = []
        for _, response_queue, _ in self.remote_envs_lists:
            profile_results.extend(response_queue.get())

        return profile_results

    def async_reset(self, seed=None):
        '''Asynchronously reset environments. Does not block.'''
        assert self.state == RESET, 'Call reset only once on initialization'
        self.state = RECV

        self.async_handles = []

        for request_queue, _, _ in self.remote_envs_lists:
            request_queue.put(("reset_all", (seed,), {}))
            if seed is not None:
                seed += self.envs_per_worker * self.binding.max_agents

    def async_call(self, name, *args, **kwargs):
        '''Asynchronously calls function environments. Does not block.'''
        self.state = RECV
        self.async_handles = []

        for request_queue, _, _ in self.remote_envs_lists:
            request_queue.put((name, args, kwargs))

    def recv(self):
        assert self.state == RECV, 'Call reset before stepping'
        self.state = SEND

        self.agent_keys = []
        obs, rewards, dones, infos = [], [], [], []
        for envs_lists in self.remote_envs_lists:
            _, response_queue, _ = envs_lists
            async_handle = response_queue.get()  # This line is blocking

            a_keys = []
            for o, r, d, i in async_handle:
                a_keys.append(list(o.keys()))
                obs += list(o.values())
                rewards += list(r.values())
                dones += list(d.values())
                infos.append(i)

            self.agent_keys.append(a_keys)

        obs = np.stack(obs)

        return obs, rewards, dones, infos

    def send(self, actions, env_id=None):
        '''Send observations to remote async environments. Does not block.'''
        assert self.state == SEND, 'Call reset + recv before send'
        self.state = RECV

        if type(actions) == list:
            actions = np.array(actions)

        actions = np.split(actions, self.num_workers)

        for envs_lists, keys_list, atns_list in zip(self.remote_envs_lists, self.agent_keys, actions):
            request_queue, response_queue, _ = envs_lists
            atns_list = np.split(atns_list, self.envs_per_worker)
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
    
    def call(self, name):
        '''Syncronously calls function remote environments. Blocks.'''
        self.async_call(name)
        all_responses = []
        for envs_lists in self.remote_envs_lists:
            _, response_queue, _ = envs_lists
            all_responses.append(response_queue.get())  # This line is blocking
        return all_responses