from pdb import set_trace as T

import ray
import numpy as np
import itertools

RESET = 0
SEND = 1
RECV = 2

def make_remote_envs(env_creator, n):
    '''Creates n serial environments on a single remote process

    Called for you by RayVecEnvs
    
    Args:
        env_creator: A function that creates a single environment
        n: The number of environments to create

    Returns:
        A Ray-backed RemoteEnvs object. Used internally by RayVecEnvs
    '''
    @ray.remote
    class RemoteEnvs:
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
        
        def reset_all(self):
            return [(e.reset(), {}, {}, {}) for e in self.envs]

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

    return RemoteEnvs.remote()


class RayVecEnv:
    def __init__(self, binding, num_workers, envs_per_worker=1):
        '''Creates env_per_worker serial environments on each of num_workers remote processes

        Synchronous API: Use env.reset() and env.step(actions)

        Asynchronous API: Use env.async_reset(), env.send(actions), and env.recv(actions)
        This confers no advantage unless you are using multiple RayVecEnvs in a double-buffered
        or multi-buffered configuration. See the PufferLib custom CleanRL demo for an example.

        Args:
            binding: A pufferlib.emulation.Binding object
            num_workers: The number of remote processes to create
            envs_per_worker: The number of serial environments to create on each remote process
        '''
        assert envs_per_worker > 0, 'Each worker must have at least 1 env'
        assert type(envs_per_worker) == int

        ray.init(
            include_dashboard=False, # WSL Compatibility
            ignore_reinit_error=True,
        )

        self.binding = binding
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker

        self.state = RESET

        self.remote_envs_lists = [
            make_remote_envs(
                binding.env_creator,
                envs_per_worker,
            ) for idx in range(num_workers)
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
        '''Not implemented yet'''
        #TODO: Implement close
        pass

    def profile(self):
        '''Returns profiling timers from all remote environments'''
        return list(itertools.chain.from_iterable(ray.get([e.profile_all.remote() for e in self.remote_envs_lists])))

    def seed(self, seed):
        '''Sets the seed for all remote environments'''
        assert type(seed) == int
        for env_list in self.remote_envs_lists:
            env_list.seed.remote(seed)
            seed += self.envs_per_worker * self.binding.max_agents

    def async_reset(self):
        '''Asynchronously reset environments. Does not block.'''
        assert self.state == RESET, 'Call reset only once on initialization'
        self.state = RECV

        self.async_handles = [e.reset_all.remote() for e in self.remote_envs_lists]

    def recv(self):
        '''Receive observations from remote async environments. Blocks.'''
        assert self.state == RECV, 'Call reset before stepping'
        self.state = SEND

        self.agent_keys = []
        obs, rewards, dones, infos = [], [], [], []
        for envs in ray.get(self.async_handles):
            a_keys = []
            for o, r, d, i in envs:
                a_keys.append(list(o.keys()))
                obs += list(o.values())
                rewards += list(r.values())
                dones += list(d.values())
                infos += list(i.values())

            self.agent_keys.append(a_keys)

        obs = np.stack(obs)

        return obs, rewards, dones, infos

    def send(self, actions, env_id=None):
        '''Send observations to remote async environments. Does not block.'''
        assert self.state == SEND, 'Call reset + recv before send'
        self.state = RECV

        #TODO: Assert size = num agents x obs
        if type(actions) == list:
            actions = np.array(actions)

        actions = np.split(actions, self.num_workers)

        self.async_handles = []
        for envs_list, keys_list, atns_list in zip(self.remote_envs_lists, self.agent_keys, actions):
            atns_list = np.split(atns_list, self.envs_per_worker)
            atns_list = [dict(zip(keys, atns)) for keys, atns in zip(keys_list, atns_list)]
            self.async_handles.append(envs_list.step.remote(atns_list))

    def reset(self):
        '''Syncronously resets remote environments. Blocks.'''
        self.async_reset()
        return self.recv()[0]

    def step(self, actions):
        '''Syncronously steps remote environments. Blocks.'''
        self.send(actions)
        return self.recv()