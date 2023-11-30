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
        os.makedirs(data_dir, exist_ok=True)

        self.policy = policy
        self.total_agents = total_agents

        # Allocate buffers
        self.actions = torch.zeros(
            total_agents, *atn_shape, dtype=int).to(device)
        self.logprobs = torch.zeros(total_agents).to(device)
        self.values = torch.zeros(total_agents).to(device)

        # Create policy sample indices and mask
        assert total_agents % len(kernel) == 0
        self.kernel = kernel * (total_agents // len(kernel))
        index_map = defaultdict(list)
        for i, elem in enumerate(self.kernel):
            index_map[elem].append(i)

        unique = np.unique(self.kernel)
        self.num_policies = len(unique)
        self.sample_idxs = [index_map[elem] for elem in unique]
        self.mask = np.zeros(total_agents)
        self.mask[self.sample_idxs[0]] = 1

        # Create ranker and storage
        self.store = pufferlib.policy_store.PolicyStore(data_dir)
        self.policy_selector = policy_selector
        self.update_policies()

        self.ranker = pufferlib.policy_ranker.Ranker(
            os.path.join(data_dir, "elo.db"))
        self.scores = {}

    def forwards(self, obs, lstm_state=None):
        policies = list(self.policies.values())

        idx = 0
        for idx in range(self.num_policies):
            samp = self.sample_idxs[idx]
            assert len(samp) > 0

            if idx == 0 or idx > len(policies):
                policy = self.policy
            else:
                policy = policies[idx - 1] # Learner not included

            # NOTE: This does not copy! Probably should.
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
        if len(infos) == self.total_agents:
            return {'learner': infos}

        policies = list(self.policies)
        policy_infos = defaultdict(list)

        idx = -1
        for game in infos:
            scores = {}
            for agent, info in game.items():
                assert isinstance(info, dict)
                idx += 1

                policy_idx = self.kernel[idx]
                if policy_idx == 0 or policy_idx > len(policies):
                    policy_name = 'learner'
                else:
                    policy_name = policies[policy_idx - 1]

                policy_infos[policy_name].append(info)

                if info_key in info:
                    scores[policy_name] = info[info_key]

            if len(scores) > 1:
                self.ranker.update(scores)

        return policy_infos

    def update_policies(self):
        policy_names = self.store.policy_names()
        selected_names = self.policy_selector(policy_names, self.num_policies-1)
        self.policies = {
            name: self.store.get_policy(name) for name in selected_names
        }
        assert len(self.policies) <= self.num_policies - 1
        logging.info(f"PolicyPool: Updated policies: {self.policies.keys()}")
