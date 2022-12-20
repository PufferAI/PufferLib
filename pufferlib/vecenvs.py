from pdb import set_trace as T

import ray
import numpy as np
import itertools


def make_remote_envs(env_creator, n):
    @ray.remote
    class RemoteEnvs:
        def __init__(self):
            self.envs = [env_creator() for _ in range(n)]
        
        def reset_all(self):
            return [e.reset() for e in self.envs]

        def step(self, actions_lists):
            all_obs, all_rewards, all_dones, all_infos = [], [], [], []

            for env, actions in zip(self.envs, actions_lists):
                if env.done:
                    obs = env.reset()
                    rewards = {k: 0 for k in obs}
                    dones = {k: False for k in obs}
                    infos = {k: {} for k in obs}
                else:
                    obs, rewards, dones, infos = env.step(actions)

                all_obs.append(obs)
                all_rewards.append(rewards)
                all_dones.append(dones)
                all_infos.append(infos)

            return all_obs, all_rewards, all_dones, all_infos

    return RemoteEnvs.remote()


class VecEnvs:
    def __init__(self, binding, num_workers, envs_per_worker=1):
        assert envs_per_worker > 0, 'Each worker must have at least 1 env'
        assert type(envs_per_worker) == int

        ray.init(
            include_dashboard=False, # WSL Compatibility
            ignore_reinit_error=True,
        )

        self.binding = binding
        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.has_reset = False

        self.remote_envs_lists = [
            make_remote_envs(
                binding.env_creator,
                envs_per_worker,
            ) for _ in range(num_workers)
        ]

        self.local_env = binding.env_creator()

    @property
    def single_observation_space(self):
        return self.binding.single_observation_space

    @property
    def single_action_space(self):
        return self.binding.single_action_space

    def _flatten(self, remote_envs_data, concat=True):
        all_keys, values = [], []
        for remote_envs in remote_envs_data:
            envs_keys = []
            for env_data in remote_envs:
                envs_keys.append(list(env_data.keys()))
                values.append(list(env_data.values()))

            all_keys.append(envs_keys)

        values = list(itertools.chain.from_iterable(values))

        if concat:
            values = np.stack(values)

        return all_keys, values

    def reset(self):
        assert not self.has_reset, 'Call reset only once on initialization'
        self.has_reset = True
        obs = ray.get([e.reset_all.remote() for e in self.remote_envs_lists])
        self.agent_keys, obs = self._flatten(obs)
        return obs

    def close(self):
        #TODO: Implement close
        pass

    def step(self, actions):
        assert self.has_reset, 'Call reset before stepping'

        #TODO: Assert size = num agents x obs
        if type(actions) == list:
            actions = np.array(actions)

        actions = np.split(actions, self.num_workers)

        rets = []
        for envs_list, keys_list, atns_list in zip(self.remote_envs_lists, self.agent_keys, actions):
            atns_list = np.split(atns_list, self.envs_per_worker)
            atns_list = [dict(zip(keys, atns)) for keys, atns in zip(keys_list, atns_list)]
            rets.append(envs_list.step.remote(atns_list))
        
        obs, rewards, dones, infos = list(zip(*ray.get(rets)))

        self.agent_keys, obs = self._flatten(obs)
        _, rewards = self._flatten(rewards)
        _, dones = self._flatten(dones)
        _, infos = self._flatten(infos, concat=False)

        return obs, rewards, dones, infos