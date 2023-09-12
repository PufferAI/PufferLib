from pdb import set_trace as T
from typing import OrderedDict, Dict

import torch
import copy

import numpy as np
import logging

from pufferlib.models import Policy

# Provides a pool of policies that collectively process a batch
# of observations. The batch is split across policies according
# to the sample weights provided at initialization.
class PolicyPool():
    def __init__(self,
        learner: Policy,
        learner_name: str,
        num_agents: int,
        num_envs: int,
        num_policies: int = 1,
        learner_weight: float = 1.0):

        self._learner = learner
        self._learner_name = learner_name
        self._learner_weight = learner_weight

        self._num_policies = num_policies
        self._policies = OrderedDict({learner_name: learner})

        self._num_agents = num_agents
        self._num_envs = num_envs
        self._batch_size = num_agents * num_envs
        self._sample_idxs = self._compute_sample_idxs()

        self.learner_mask = np.zeros(self._batch_size)
        self.learner_mask[self._sample_idxs[0]] = 1
        self.scores = {}

        self._allocated = False

    def _compute_sample_idxs(self):
        # Create indices for splitting data across policies
        ow = 0
        if self._num_policies > 1:
            ow = int(self._num_agents * (1 - self._learner_weight) / (self._num_policies - 1))

        lw = self._num_agents - ow * (self._num_policies - 1)

        sample_weights = [lw] + [ow] * (self._num_policies - 1)
        print(f"PolicyPool sample_weights: {sample_weights}")
        chunk_size = sum(sample_weights)
        pattern = [i for i, weight in enumerate(sample_weights)
                for _ in range(weight)]

        # Distribute indices among sublists
        sample_idxs = [[] for _ in range(self._num_policies)]
        for idx in range(self._batch_size):
            sublist_idx = pattern[idx % chunk_size]
            sample_idxs[sublist_idx].append(idx)

        return sample_idxs

    def forwards(self, obs, lstm_state=None, dones=None):
        batch_size = len(obs)
        for samp, policy in zip(self._sample_idxs, self._policies.values()):
            if len(samp) == 0:
                continue
            if lstm_state is not None:
                atn, lgprob, _, val, (lstm_state[0][:, samp], lstm_state[1][:, samp]) = policy.get_action_and_value(
                    obs[samp],
                    state=[lstm_state[0][:, samp], lstm_state[1][:, samp]],
                    done=dones[samp])
            else:
                atn, lgprob, _, val = policy.get_action_and_value(obs[samp])

            if not self._allocated:
                self._allocated = True

                self.actions = torch.zeros(batch_size, *atn.shape[1:], dtype=int).to(atn.device)
                self.logprobs = torch.zeros(batch_size).to(lgprob.device)
                self.values = torch.zeros(batch_size).to(val.device)

            self.actions[samp] = atn
            self.logprobs[samp] = lgprob
            self.values[samp] = val.flatten()

        return self.actions, self.logprobs, self.values, lstm_state

    def update_scores(self, infos, info_key):
        # TODO: Check that infos is dense and sorted
        agent_infos = []
        if self._num_agents > 1:
            for info in infos:
                agent_infos += list(info.values())
        else:
            agent_infos = infos

        policy_infos = {}
        for samp, (name, policy) in zip(self._sample_idxs, self._policies.items()):
            pol_infos = np.array(agent_infos)[samp]
            policy_infos[name] = pol_infos

            for i in pol_infos:
                if info_key not in i:
                    continue

                if name not in self.scores:
                    self.scores[name] = []

                self.scores[name].append(i[info_key])

        return policy_infos

    # Update the active policies to be used for the next batch. Always
    # include the required policies, and then randomly sample the rest
    # from the available policies.
    def update_policies(self, policies: Dict[str, Policy]):
        self._policies = {self._learner_name: self._learner, **policies}
        logging.info(f"PolicyPool: Updated policies: {self._policies.keys()}")
