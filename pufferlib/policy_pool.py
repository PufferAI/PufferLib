from pdb import set_trace as T
import numpy as np

from collections import defaultdict

import torch
import copy
import logging
import os

from pufferlib.models import Policy
import pufferlib.policy_store
import pufferlib.policy_ranker
import pufferlib.utils


def random_selector(items: list, num: int):
    if len(items) == 0:
        return []

    return np.random.choice(items, num, replace=True).tolist()

class PolicyPool:
    def __init__(self, policy, total_agents, atn_shape, device,
            data_dir='experiments', kernel=[0],
            policy_selector: callable = random_selector):
        '''Provides a pool of policies that collectively process a batch
        of observations. The batch is split across policies according
        to the sample weights provided at initialization.'''
        assert data_dir is not None
        self.policy = policy
        self.total_agents = total_agents

        # Allocate buffers
        self.actions = torch.zeros(
            total_agents, *atn_shape, dtype=int).to(device)
        self.logprobs = torch.zeros(total_agents).to(device)
        self.values = torch.zeros(total_agents).to(device)

        # Create policy sample indices and mask
        assert total_agents % len(kernel) == 0
        kernel = kernel * (total_agents // len(kernel))
        index_map = defaultdict(list)
        for i, elem in enumerate(kernel):
            index_map[elem].append(i)

        unique = np.unique(kernel)
        self.num_policies = len(unique)
        self.sample_idxs = [index_map[elem] for elem in unique]
        self.mask = np.zeros(total_agents)
        self.mask[self.sample_idxs[0]] = 1

        # Create ranker and storage
        self.ranker = pufferlib.policy_ranker.Ranker(
            os.path.join(data_dir, "elo.db"))
        self.scores = {}

        self.store = pufferlib.policy_store.PolicyStore(
            os.path.join(data_dir, "policies"))
        self.policy_selector = policy_selector
        self.update_policies()


    def forwards(self, obs, lstm_state=None):
        policies = list(self.policies.values())

        idx = 0
        for idx in range(self.num_policies):
            samp = self.sample_idxs[idx]
            assert len(samp) > 0

            missing = idx >= len(policies)
            policy = self.policy if missing else policies[idx]

            if lstm_state is not None:
                h = lstm_state[0][:, samp]
                c = lstm_state[1][:, samp]
                atn, lgprob, _, val, (h, c) = policy.get_action_and_value(
                    obs[samp],
                    (h, c),
                  )
                lstm_state[0][:, samp] = h
                lstm_state[1][:, samp] = c
            else:
                atn, lgprob, _, val = policy.get_action_and_value(obs[samp])

            self.actions[samp] = atn
            self.logprobs[samp] = lgprob
            self.values[samp] = val.flatten()

        return self.actions, self.logprobs, self.values, lstm_state

    def update_scores(self, infos, info_key):
        # TODO: Check that infos is dense and sorted
        if len(infos) != self.total_agents:
            agent_infos = []
            for info in infos:
                agent_infos += list(info.values())
        else:
            agent_infos = infos

        policy_infos = {}
        for samp, (name, policy) in zip(self.sample_idxs, self.policies.items()):
            pol_infos = np.array(agent_infos)[samp]
            policy_infos[name] = pol_infos

            for i in pol_infos:
                if info_key not in i:
                    continue

                if name not in self.scores:
                    self.scores[name] = []

                self.scores[name].append(i[info_key])

        return policy_infos

    def update_ranks(self):
        if self.scores:
            self.policy_ranker.update(
                self.policy_pool.scores,
            )
            self.scores = {}

    def update_policies(self):
        policy_names = self.store.policy_names()
        selected_names = self.policy_selector(policy_names, self.num_policies-1)
        self.policies = {
            name: self.store.get_policy(name) for name in selected_names
        }
        logging.info(f"PolicyPool: Updated policies: {self.policies.keys()}")



