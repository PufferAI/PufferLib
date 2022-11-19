from pdb import set_trace as T

import ray

import torch
from torch.distributions import Categorical

import numpy as np
import itertools

from pufferlib.frameworks import make_recurrent_policy, BasePolicy

def make_remote_envs(env_cls, env_args, n):
    @ray.remote
    class RemoteEnvs:
        def __init__(self):
            self.envs = [env_cls(env_args) for _ in range(n)]
        
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
                binding.env_cls,
                binding.env_args,
                envs_per_worker,
            ) for _ in range(num_workers)
        ]

    @property
    def single_observation_space(self):
        return self.binding.observation_space

    @property
    def single_action_space(self):
        return self.binding.action_space

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

    def step(self, actions):
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


def make_cleanrl_policy(policy_cls, lstm_layers=0):
    assert issubclass(policy_cls, BasePolicy)

    class CleanRLPolicy(policy_cls):
        '''Temporary hack to get framework running with CleanRL

        Their LSTMs are kind of weird. Need to figure this out'''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if lstm_layers > 0:
                self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, lstm_layers)

        # TODO: Cache value
        def get_value(self, x, lstm_state=None, done=None):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
                #lstm_state = [lstm_state[0].unsqueeze(1), lstm_state[1].unsqueeze(1)]

            hidden, _ = self.encode_observations(x)
            hidden, lstm_state = self._compute_lstm(hidden, lstm_state, done)

            return self.value_head(hidden)

        # TODO: Compute seq_lens from done, replace with PufferLib LSTM
        def _compute_lstm(self, hidden, lstm_state, done):
            batch_size = lstm_state[0].shape[1]
            hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
            done = done.reshape((-1, batch_size))
            new_hidden = []
            for h, d in zip(hidden, done):
                h, lstm_state = self.lstm(
                    h.unsqueeze(0),
                    (
                        (1.0 - d).view(1, -1, 1) * lstm_state[0],
                        (1.0 - d).view(1, -1, 1) * lstm_state[1],
                    ),
                )
                new_hidden += [h]
            return torch.flatten(torch.cat(new_hidden), 0, 1), lstm_state

        # TODO: Compute seq_lens from done
        def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            hidden, _ = self.encode_observations(x)
            hidden, lstm_state = self._compute_lstm(hidden, lstm_state, done)

            value = self.value_head(hidden)
            flat_logits = self.decode_actions(hidden, None, concat=False)

            multi_categorical = [Categorical(logits=l) for l in flat_logits]

            if action is None:
                action = torch.stack([c.sample() for c in multi_categorical])
            else:
                action = action.view(-1, action.shape[-1]).T

            logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
            entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)

            if lstm_layers > 0:
                return action.T, logprob, entropy, value, lstm_state
            return action.T, logprob, entropy, value
   
    return CleanRLPolicy