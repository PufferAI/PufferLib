from pdb import set_trace as T
import numpy as np

from typing import OrderedDict, Dict
from collections import defaultdict

import torch
import copy

import logging

from pufferlib.models import Policy

import numpy as np


def index_groups(inp):
    index_map = defaultdict(list)
    
    for i, elem in enumerate(inp):
        index_map[elem].append(i)
    
    unique_elements = np.unique(inp)
    return [index_map[elem] for elem in unique_elements]

# Provides a pool of policies that collectively process a batch
# of observations. The batch is split across policies according
# to the sample weights provided at initialization.
class PolicyPool():
    def __init__(self,
            learner: Policy,
            learner_name: str,
            num_agents: int,
            num_envs: int,
            kernel: list = None,
            ):
        self._learner = learner
        self._learner_name = learner_name
        
        if kernel is None:
            kernel = [0]

        self.kernel = kernel
        self._num_policies = len(np.unique(kernel))
        self._policies = OrderedDict({learner_name: learner})

        self._num_agents = num_agents
        self._num_envs = num_envs
        self._batch_size = num_agents * num_envs

        # Used to distribute the batch across policies
        assert self._batch_size % len(self.kernel) == 0
        self._sample_idxs = index_groups(
            kernel * (self._batch_size // len(kernel)))

        self.learner_mask = np.zeros(self._batch_size)
        self.learner_mask[self._sample_idxs[0]] = 1
        self.scores = {}

        self._allocated = False

    def forwards(self, obs, lstm_state=None):
        batch_size = len(obs)
        policies = list(self._policies.values())
        idx = 0

        for idx in range(self._num_policies):
            samp = self._sample_idxs[idx]
            assert len(samp) > 0
            if idx >= len(policies):
                policy = self._learner
            else:
                policy = policies[idx]

            if lstm_state is not None:
                atn, lgprob, _, val, (lstm_state[0][:, samp], lstm_state[1][:, samp]) = policy.get_action_and_value(
                    obs[samp],
                    state=[lstm_state[0][:, samp], lstm_state[1][:, samp]],
                  )
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
