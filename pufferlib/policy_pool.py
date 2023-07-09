from functools import lru_cache
from pdb import set_trace as T
from collections import defaultdict

import torch
import copy

import numpy as np
import pandas as pd

from pufferlib.models import Policy

# Provides a pool of policies that collectively process a batch
# of observations. The batch is split across policies according
# to the sample weights provided at initialization.
class PolicyPool():
    def __init__(self, learner: Policy, learner_weight: float):
        self._learner = learner
        self._learner_weight = learner_weight

        self._scores = defaultdict(list)
        self._num_scores = 0
        self._policies = [learner]

    @lru_cache(maxsize=1)
    def _sample_indxs(self, batch_size):
        num_policies = len(self._policies)
        learner_batch = int(batch_size * self._learner_weight)

        other_batch = 0
        if num_policies > 1:
            other_batch = (batch_size - learner_batch) // (num_policies - 1)

        # Create indices for splitting data across policies
        sample_weights = [learner_batch] + [other_batch for _ in self._policies]
        chunk_size = sum(sample_weights)
        pattern = [i for i, weight in enumerate(sample_weights)
                for _ in range(weight)]

        # Distribute indices among sublists
        sample_idxs = [[] for _ in range(num_policies)]
        for idx in range(batch_size):
            sublist_idx = pattern[idx % chunk_size]
            sample_idxs[sublist_idx].append(idx)
        return sample_idxs

    def forwards(self, obs, lstm_state=None, dones=None):
        batch_size = len(obs)
        for samp, policy in zip(self._sample_idxs(), self._policies):
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

        policy_infos = {}
        for samp, policy in zip(self.sample_idxs, self.active_policies):
            pol_infos = np.array(agent_infos)[samp]
            if policy.name not in policy_infos:
                policy_infos[policy.name] = list(pol_infos)
            else:
                policy_infos[policy.name] += list(pol_infos)

            for i in pol_infos:
                if info_key not in i:
                    continue

                self.scores[policy.name].append(i[info_key])
                self.num_scores += 1

        return policy_infos

    # Update the active policies to be used for the next batch. Always
    # include the required policies, and then randomly sample the rest
    # from the available policies.
    def update_active_policies(self, policies):
        if required_policy_names is None:
            required_policy_names = []

        num_needed = self._num_active_policies - len(required_policy_names)
        new_policy_names = required_policy_names + \
        self._policy_selector.select_policies(num_needed, exclude=required_policy_names)

        new_policies = OrderedDict()
        for policy_name in new_policy_names:
            new_policies[policy_name] = self._loaded_policies.get(
                policy_name,
                self._policy_loader.load_policy(policy_name))
        self._policies = list(new_policies.values())
        self._loaded_policies = new_policies
