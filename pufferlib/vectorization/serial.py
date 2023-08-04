from pdb import set_trace as T

import numpy as np
import itertools

RESET = 0
SEND = 1
RECV = 2


class VecEnv:
    def __init__(self, binding, num_workers, envs_per_worker=1):
        assert envs_per_worker > 0, 'Each worker must have at least 1 env'
        assert type(envs_per_worker) == int

        self.binding = binding
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker

        self.state = RESET

        self.envs_lists = [
            make_serial_envs(
                self.binding.env_creator,
                self.envs_per_worker,
            ) for idx in range(self.num_workers)
        ]

    @property
    def single_observation_space(self):
        return self.binding.single_observation_space

    @property
    def single_action_space(self):
        return self.binding.single_action_space

    def close(self):
        pass

    def profile(self):
        return list(itertools.chain.from_iterable([e.profile_all() for e in self.envs_lists]))

    def async_reset(self, seed=None):
        assert self.state == RESET, 'Call reset only once on initialization'
        self.state = RECV

        self.async_handles = []

        for e in self.envs_lists:
            self.async_handles.append(e.reset_all(seed=seed))
            if seed is not None:
                seed += self.envs_per_worker * self.binding.max_agents

    def recv(self):
        assert self.state == RECV, 'Call reset before stepping'
        self.state = SEND

        self.agent_keys = []
        obs, rewards, dones, infos = [], [], [], []
        for envs in self.async_handles:
            a_keys = []
            for o, r, d, i in envs:
                a_keys.append(list(o.keys()))
                obs += list(o.values())
                rewards += list(r.values())
                dones += list(d.values())
                infos.append(i)

            self.agent_keys.append(a_keys)

        obs = np.stack(obs)

        return obs, rewards, dones, infos

    def send(self, actions, env_id=None):
        assert self.state == SEND, 'Call reset + recv before send'
        self.state = RECV

        if type(actions) == list:
            actions = np.array(actions)

        actions = np.split(actions, self.num_workers)

        self.async_handles = []
        for envs_list, keys_list, atns_list in zip(self.envs_lists, self.agent_keys, actions):
            atns_list = np.split(atns_list, self.envs_per_worker)
            atns_list = [dict(zip(keys, atns)) for keys, atns in zip(keys_list, atns_list)]
            self.async_handles.append(envs_list.step(atns_list))

    def reset(self, seed=None):
        self.async_reset()
        return self.recv()[0]

    def step(self, actions):
        self.send(actions)
        return self.recv()


def make_serial_envs(env_creator, n):
    class SerialEnvs:
        def __init__(self):
            self.envs = [env_creator() for _ in range(n)]

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
            for e in self.envs:
                obs = e.reset(seed=seed) 
                async_handles.append((obs, {}, {}, {k: {} for k in obs}))
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
                    infos = {k: {} for k in obs}
                else:
                    obs, rewards, dones, infos = env.step(actions)

                returns.append((obs, rewards, dones, infos))

            return returns

    return SerialEnvs()