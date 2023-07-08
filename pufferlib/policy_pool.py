from pdb import set_trace as T
from collections import defaultdict

import torch
import copy

import numpy as np
import pandas as pd

# Provides a pool of policies that collectively process a batch
# of observations. The batch is split across policies according
# to the sample weights provided at initialization.
class PolicyPool():
    def __init__(self, batch_size, sample_weights):

        assert len(sample_weights) == active_policies

        self.learner = learner
        self.learner_name = name

        # Set up skill rating tournament
        self.tournament = OpenSkillRating(mu, anchor_mu, sigma)
        self.scores = defaultdict(list)
        self.mu = mu
        self.anchor_mu = anchor_mu
        self.sigma = sigma

        self.num_scores = 0
        self.num_active_policies = active_policies
        self.active_policies = []
        self.path = path

        # Set up the SQLite database and session
        self.database = PolicyDatabase()

        # Assign policies used for evaluation
        self.add_policy(learner, name, tenured=True, mu=mu, sigma=sigma, anchor=False)
        self.update_active_policies()

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

    @property
    def ratings(self):
        return self.tournament.ratings

    def add_policy_copy(self, key, name, tenured=False, anchor=False):
        # Retrieve the policy from the database using the key
        original_policy = self.database.query_policy_by_name(key)
        assert original_policy is not None, f"Policy with name '{key}' does not exist."

        # Use add_policy method to add the new policy
        self.add_policy(original_policy.model, name, tenured=tenured, mu=original_policy.mu, sigma=original_policy.sigma, anchor=anchor)

    def add_policy(self, model, name, tenured=False, mu=None, sigma=None, anchor=False, overwrite_existing=True):
        # Construct the model path by joining the model and name
        model_path = f"{self.path}/{name}"

        # Check if a policy with the same name already exists in the database
        existing_policy = self.database.query_policy_by_name(name)

        if existing_policy is not None:
            if overwrite_existing:
                self.database.delete_policy(existing_policy)
            else:
                raise ValueError(f"A policy with the name '{name}' already exists.")

        # Set default values for mu and sigma if they are not provided
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma

        # TODO: Eliminate need to deep copy
        model = copy.deepcopy(model)
        policy = Policy(
            model=model,
            model_path=model_path,
            model_class=str(type(model)),
            name=name,
            mu=mu,
            sigma=sigma,
            episodes=0,  # assuming new policies have 0 episodes
            additional_data={'tenured': tenured}
        )
        policy.save_model(model)

        # Add the new policy to the database
        self.database.add_policy(policy)

        # Add the policy to the tournament system
        # TODO: Figure out anchoring
        if anchor:
            self.tournament.set_anchor(name)
        else:
            self.tournament.add_policy(name)
            self.tournament.ratings[name].mu = mu
            self.tournament.ratings[name].sigma = sigma

    def forwards(self, obs, lstm_state=None, dones=None):
        batch_size = len(obs)
        for samp, policy in zip(self.sample_idxs, self.active_policies):
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
        self._active_policies = list(new_policies.values())
        self._loaded_policies = new_policies
