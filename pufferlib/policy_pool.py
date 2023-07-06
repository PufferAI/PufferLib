from pdb import set_trace as T
from collections import defaultdict

import torch
import copy

import numpy as np
import pandas as pd

from pufferlib.rating import OpenSkillRating

# Provides a pool of policies that collectively process a batch
# of observations. The batch is split across policies according
# to the sample weights provided at initialization.
class PolicyPool():
    def __init__(self, batch_size, sample_weights):

        self._active_policies = []
        self._sample_weights = sample_weights
        self._num_active_policies = len(sample_weights)

        # Create indices for splitting data across policies
        chunk_size = sum(sample_weights)
        assert batch_size % chunk_size == 0
        pattern = [i for i, weight in enumerate(sample_weights)
                for _ in range(weight)]

        # Distribute indices among sublists
        self._sample_idxs = [[] for _ in range(self._num_active_policies)]
        for idx in range(batch_size):
            sublist_idx = pattern[idx % chunk_size]
            self._sample_idxs[sublist_idx].append(idx)

    def forwards(self, obs, lstm_state=None, dones=None):
        all_actions = None
        returns = []
        for samp, policy in zip(self._sample_idxs, self._active_policies):
            if lstm_state is not None:
                atn, lgprob, _, val, (lstm_state[0][:, samp], lstm_state[1][:, samp]) = policy.model.get_action_and_value(
                    obs[samp],
                    [lstm_state[0][:, samp], lstm_state[1][:, samp]],
                    dones[samp])
            else:
                atn, lgprob, _, val = policy.model.get_action_and_value(obs[samp])

            if all_actions is None:
                all_actions = torch.zeros((len(obs), *atn.shape[1:]), dtype=atn.dtype).to(atn.device)

            returns.append((atn, lgprob, val, lstm_state, samp))
            all_actions[samp] = atn

        return all_actions, returns

    def update_scores(self, infos, info_key):
        # TODO: Check that infos is dense and sorted
        agent_infos = []
        for info in infos:
            agent_infos += list(info.values())

        policy_infos = defaultdict(list)
        for samp, policy in zip(self._sample_idxs, self._active_policies):
            pol_infos = np.array(agent_infos)[samp]
            policy_infos[policy.name] += list(pol_infos)

            for i in pol_infos:
                if info_key not in i:
                    continue

                self.scores[policy.name].append(i[info_key])
                self.num_scores += 1

        return policy_infos
