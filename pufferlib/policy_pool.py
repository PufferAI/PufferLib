from pdb import set_trace as T
from collections import defaultdict

import torch
import copy
import openskill

import numpy as np
import pandas as pd
from pufferlib.rating import OpenSkillRating


class PolicyPoolRecord():
    def __init__(self, model, name, path, mu, sigma, tenured, episodes=0):
        self.model = model
        self.path = path
        self.name = name

        self.mu = mu
        self.sigma = sigma

        self.tenured = tenured
        self.episodes = episodes

        self._save()

    def update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self._save()

    def _save(self):
        torch.save(self.model.state_dict(), self.path + '.pt')
        np.save(self.path + '.npy', {
            'episodes': self.episodes,
            'mu': self.mu,
            'sigma': self.sigma
        })

    @classmethod
    def load(cls, path):
        model = torch.load(path + '.pt')
        data = np.load(path + '.npy')

        return cls(model, path, data['mu'],
                data['sigma'], data['episodes'])

# Maintains a pool of policies, and allows random sampling from the pool
# based on the mean reward of each policy.
class PolicyPool():
    def __init__(self, evaluation_batch_size, policies=[], names=[], 
            tenured=[], sample_weights=[], max_policies=1,
            path='pool', rank_update_scores=10, policy_update_scores=100,
            mu=1000, anchor_mu=1000, sigma=100/3):
        self.tournament = OpenSkillRating(mu, anchor_mu, sigma)
        self.policies = {}
        self.active_policies = {}
        self.path = path

        self.scores = defaultdict(list)
        self.num_scores = 0
        self.rank_update_scores = rank_update_scores
        self.policy_update_scores = policy_update_scores
        
        self.mu = mu
        self.anchor_mu = anchor_mu
        self.sigma = sigma

        self.num_active_policies = len(sample_weights)
        self.max_policies = max_policies

        # Assign policies used for evaluation
        for p, n, t in zip(policies, names, tenured):
            self.add_policy(p, n, tenured=t)
        self.update_active_policies()

        # Create indices for splitting data across policies
        chunk_size = sum(sample_weights)
        assert evaluation_batch_size % chunk_size == 0
        pattern = [i for i, weight in enumerate(sample_weights)
                for _ in range(weight)]

        # Distribute indices among sublists
        self.sample_idxs = [[] for _ in range(len(sample_weights))]
        for idx in range(evaluation_batch_size):
            sublist_idx = pattern[idx % chunk_size]
            self.sample_idxs[sublist_idx].append(idx)

    def forwards(self, obs, lstm_state=None, dones=None):
        returns = []
        all_actions = []
        policy_keys = list(self.active_policies.keys())
        for idx in range(self.num_active_policies):
            key = policy_keys[0]
            if idx < len(self.active_policies):
                key = policy_keys[idx]

            policy = self.active_policies[key]
            samp = self.sample_idxs[idx]
            if lstm_state is not None:
                atn, lgprob, _, val, (lstm_state[0][:, samp], lstm_state[1][:, samp]) = policy.model.get_action_and_value(
                    obs[samp],
                    [lstm_state[0][:, samp], lstm_state[1][:, samp]],
                    dones[samp])
            else:
                atn, lgprob, _, val = policy(obs[samp])
            
            returns.append((atn, lgprob, val, lstm_state))
            all_actions.append(atn)
        
        return torch.cat(all_actions), self.sample_idxs, returns

    def load(self, path):
        '''Load all models in path'''
        for idx in range(self._max_policies):
            policy = PolicyPoolRecord.load(path + f'/{idx}')
            self.policies.add(policy)

    def add_policy_copy(self, key, name, tenured=False, anchor=False):
        policy = self.policies[key]
        model = copy.deepcopy(policy.model)
        self.add_policy(model, name, tenured, mu=policy.mu, sigma=policy.sigma, anchor=anchor)

    def add_policy(self, model, name, tenured=False, mu=None, sigma=None, anchor=False):
        assert name not in self.policies, 'Policy name already exists'

        if mu is None:
            mu = self.mu

        if sigma is None:
            sigma = self.sigma

        policy = PolicyPoolRecord(model, name, self.path,
                mu, sigma, tenured )

        # TODO: Figure out anchoring
        if anchor:
            self.tournament.set_anchor(name)
        else:
            self.tournament.add_policy(name)
            self.tournament.ratings[name].mu = mu
            self.tournament.ratings[name].sigma = sigma

        if len(self.policies) == self.max_policies:
            worst_score = float('inf')
            for name, policy in self.policies.items():
                if not policy.tenured and policy.mu < worst_score:
                    worst_policy = name
                    worst_score = policy.mu

            del self.policies[worst_policy]

        self.policies[name] = policy

    def update_scores(self, infos, key):
        # TODO: Check that infos is dense and sorted
        agent_infos = []
        for info in infos:
            agent_infos += list(info.values())

        policy_keys = list(self.active_policies.keys())
        for idx in range(self.num_active_policies):
            if idx >= len(self.active_policies):
                idx = 0

            update_policies = False
            update_ranks = False
            samp = self.sample_idxs[idx]
            policy = policy_keys[idx]
            for i in np.array(agent_infos)[samp]:
                if key not in i:
                    continue 
            
                self.scores[policy].append(i[key])
                self.num_scores += 1

                if self.num_scores % self.policy_update_scores == 0:
                    update_policies = True

                if self.num_scores % self.rank_update_scores == 0:
                    update_ranks = True

        if update_ranks:
            self.update_ranks()

        if update_policies:
            self.update_active_policies()

    def update_ranks(self):
        self.tournament.update(
            list(self.scores.keys()),
            list(self.scores.values())
        )

        for name, rating in self.tournament.ratings.items():
            if name in self.policies:
                self.policies[name].mu = rating.mu
                self.policies[name].sigma = rating.sigma

        self.scores = defaultdict(list)

    def update_active_policies(self):
        active_policies = [k for k, v in self.policies.items() if v.tenured]
        untenured = [k for k, v in self.policies.items() if not v.tenured]

        # Randomly sample from untenured policies
        if len(untenured) > 0:
            active_policies += np.random.choice(untenured,
                   self.num_active_policies - len(active_policies),
                   replace=True).tolist()

        self.active_policies = {k: self.policies[k] for k in active_policies}

    def to_table(self):
        stats = self.tournament.stats
        table = pd.DataFrame(self.policies.keys(), columns=["Model"])
        table["Rank"] = [stats[model] for model in table["Model"]]
        table["Num Samples"] = [self._policies[model].num_samples() for model in table["Model"]]
        table['Experiment'] = table['Model'].apply(lambda x: x.split('/')[-2])
        table['Checkpoint'] = table['Model'].apply(lambda x: int(x.split('/')[-1].split('.')[0]))

        table = table.sort_values(by='Rank', ascending=False)
        return table
