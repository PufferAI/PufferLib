from pdb import set_trace as T
from collections import defaultdict
import logging
import os

import numpy as np
import torch

import pufferlib.policy_store
from pufferlib.policy_store import get_policy_names  # provide as a helper function
import pufferlib.policy_ranker
import pufferlib.utils

LEARNER_POLICY_ID = 0


# Kernel helper
def create_kernel(agents_per_env, num_policies,
                  shuffle_with_seed=None):
    assert agents_per_env > 0 and num_policies >= 0, \
        'create_kernel: agents_per_env and num_policies must be non-negative'
    if num_policies == 0:
        return [LEARNER_POLICY_ID]
    if num_policies == 1:  # probably, the PvE case
        return [1]  # will be tiled to all agents

    agents_per_agents = agents_per_env // num_policies
    kernel = []
    for i in range(num_policies):
        kernel.extend([i+1] * agents_per_agents)  # policies are 1-indexed
    kernel.extend([LEARNER_POLICY_ID] * (agents_per_env - len(kernel)))

    if isinstance(shuffle_with_seed, int):  # valid seed
        rng = np.random.RandomState(shuffle_with_seed)
        rng.shuffle(kernel)

    return kernel

# Policy selectors
class PolicySelector:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, items: list, num: int):
        raise NotImplementedError

class RandomPolicySelector(PolicySelector):
    def __call__(self, items: list, num: int):
        assert len(items) > 0, 'RandomPolicySelector: items must be non-empty'
        return self.rng.choice(items, num, replace=len(items) < num).tolist()

class AllPolicySelector(PolicySelector):
    def __call__(self, items: list, num: int):
        assert len(items) == num, 'AllPolicySelector: num must match len(items)'
        return self.rng.permutation(items).tolist()

def random_selector(items: list, num: int):
    logging.warning('random_selector: This breaks determinism. Use RandomPolicySelector instead.')
    assert len(items) > 0, 'random_selector: items must be non-empty'
    # allow replacement if there are fewer items than requested
    return np.random.choice(items, num, replace=len(items) < num).tolist()

class PolicyPool:
    def __init__(self, policy, total_agents, atn_shape, device,
            data_dir=None,
            kernel=[LEARNER_POLICY_ID],
            policy_selector: callable = None,
            policy_store: pufferlib.policy_store.PolicyStore=None,
            skip_ranker=False,  # for testing
        ):
        '''Provides a pool of policies that collectively process a batch
        of observations. The batch is split across policies according
        to the sample weights provided at initialization.'''
        self.learner_policy = policy
        self.total_agents = total_agents

        # Allocate buffers
        self.actions = torch.zeros(
            total_agents, *atn_shape, dtype=int).to(device)
        self.logprobs = torch.zeros(total_agents).to(device)
        self.values = torch.zeros(total_agents).to(device)

        # Create sample_idxs from kernel
        self.policy_ids, self.sample_idxs, self.kernel = \
            self._init_sample_idx_from_kernel(kernel)

        # Create learner mask
        self.mask = np.zeros(total_agents)
        if LEARNER_POLICY_ID in self.sample_idxs:
            self.mask[self.sample_idxs[LEARNER_POLICY_ID]] = 1

        # Create policy store, selector, and initial current_policies
        if policy_store is None:
            assert data_dir is not None, 'PolicyPool: Must provide data_dir'
            os.makedirs(data_dir, exist_ok=True)
            self.store = pufferlib.policy_store.PolicyStore(data_dir)
        else:
            self.store = policy_store
            data_dir = self.store.path

        self.policy_selector = policy_selector or RandomPolicySelector(seed=0)
        self.current_policies = {}  # Dict[policy_id] = (name, Policy)

        # Create ranker
        self.ranker = None
        if not skip_ranker:
            self.ranker = pufferlib.policy_ranker.Ranker(
                os.path.join(data_dir, 'elo.db'))
        self.scores = None

        # Load policies using the policy store and selector
        self.update_policies()

    def _init_sample_idx_from_kernel(self, kernel):
        # Made this a method so it can be tested
        assert self.total_agents % len(kernel) == 0, 'Kernel does not divide agents'
        policy_ids = np.unique(kernel)
        kernel = kernel * (self.total_agents // len(kernel))

        index_map = defaultdict(list)
        for i, elem in enumerate(kernel):
            index_map[elem].append(i)
        sample_idxs = {elem: index_map[elem] for elem in policy_ids}

        return policy_ids, sample_idxs, kernel

    def forwards(self, obs, lstm_state=None):
        for policy_id in self.policy_ids:
            samp = self.sample_idxs[policy_id]
            assert len(samp) > 0

            policy = self.learner_policy if policy_id == LEARNER_POLICY_ID \
                else self.current_policies[policy_id]['policy']

            # NOTE: This does not copy! Probably should.
            if lstm_state is not None:
                h = lstm_state[0][:, samp]
                c = lstm_state[1][:, samp]
                atn, lgprob, _, val, (h, c) = policy(obs[samp], (h, c))
                lstm_state[0][:, samp] = h
                lstm_state[1][:, samp] = c
            else:
                atn, lgprob, _, val = policy(obs[samp])

            self.actions[samp] = atn
            self.logprobs[samp] = lgprob
            self.values[samp] = val.flatten()

        return self.actions, self.logprobs, self.values, lstm_state

    def _get_policy_name(self, agent_id):
        assert agent_id > 0, 'Agent id must be > 0'
        policy_id = self.kernel[agent_id-1]  # agent_id is 1-indexed
        policy_name = 'learner' if policy_id == LEARNER_POLICY_ID \
            else self.current_policies[policy_id]['name']
        return policy_name

    def update_scores(self, infos, score_key):
        if len(infos) == self.total_agents \
          and np.all(self.policy_ids == LEARNER_POLICY_ID):
            return {'learner': infos}

        policy_infos = defaultdict(list)

        for game in infos:
            agent_scores = defaultdict(list)
            for agent_id, info in game.items():
                assert isinstance(info, dict)
                assert agent_id > 0, 'Agent id must be > 0'

                policy_name = self._get_policy_name(agent_id)
                policy_infos[policy_name].append(info)

                if score_key in info:
                    agent_scores[policy_name].append(info[score_key])

            self.scores = {name: np.mean(s) for name, s in agent_scores.items()}
            if self.ranker is not None and len(self.scores) > 1:
                self.ranker.update(self.scores)

        return policy_infos

    def update_policies(self,
                        policy_ids=None,  # these args are for testing
                        store=None,
                        policy_selector=None):
        if policy_ids is None:
            policy_ids = self.policy_ids
        num_policies_to_select = sum(policy_ids > LEARNER_POLICY_ID)
        if num_policies_to_select == 0:
            return

        self.current_policies.clear()

        store = store or self.store
        policy_names = store.policy_names()
        if len(policy_names) == 0:
            # fill all poclies with the learner policy
            for policy_id in policy_ids:
                self.current_policies[policy_id] = {
                    'name': 'learner',
                    'policy': self.learner_policy
                }
        else:
            policy_selector = policy_selector or self.policy_selector
            selected_names = policy_selector(policy_names, num_policies_to_select)

            for policy_id in policy_ids:
                if policy_id > LEARNER_POLICY_ID:
                    name = selected_names.pop(0)
                    self.current_policies[policy_id] = {
                        'name': name,
                        'policy': store.get_policy(name),
                    }

            if self.ranker is not None and self.scores is not None:
                logging.info(f'PolicyPool: Score board\n{self.ranker}')

            logging.info(f"""Loaded policies: {[
                p['name'] for p in self.current_policies.values()
            ]}\n""")
