import unittest

import numpy as np
import torch

import pufferlib.policy_pool as pp

NUM_AGENTS = 4
NUM_ENVS = 2
POOL_AGENTS = NUM_AGENTS * NUM_ENVS  # batch size
OBS_DIM = 3
ACTION_DIM = 5

# TODO: add test for recurrent policy forward
BPTT_HORIZON = 16
LSTM_INPUT_DIM = POOL_AGENTS * BPTT_HORIZON
LSTM_HIDDEN_DIM = 32


class MockPolicy:
    def __call__(self, obs):
        batch_size = obs.shape[0]
        actions = torch.arange(batch_size * ACTION_DIM).view(batch_size, ACTION_DIM)
        logprobs = torch.arange(batch_size, dtype=torch.float32)
        values = torch.arange(batch_size, dtype=torch.float32) + 10  # add to make the values different
        return actions, logprobs, None, values

class MockPolicyStore:
    def __init__(self, num_policies):
        self._policies = {f'Policy{i+1}': MockPolicy() for i in range(num_policies)}
        self.path = 'mock_policy_store'

    def policy_names(self):
        return list(self._policies.keys())

    def get_policy(self, name):
        return self._policies[name]

class TestPolicyPool(unittest.TestCase):
    def setUp(self):
        self.mock_nonrecurrent_policy = MockPolicy()
        self.mock_nonrecurrent_policy.name = 'BasePolicy1'
        self.nonrecurrent_policy_pool = pp.PolicyPool(
            policy=self.mock_nonrecurrent_policy,
            total_agents=POOL_AGENTS,
            atn_shape=(ACTION_DIM,),
            device='cpu',
            policy_store=MockPolicyStore(3),
            kernel = [0, 1, 0, 2],
            skip_ranker=True,
        )

    def test_init_with_kernel(self):
        test_policy_pool = self.nonrecurrent_policy_pool
        kernel = [0, 1, 0, 2]
        policy_ids, sample_idxs, kernel = test_policy_pool._init_sample_idx_from_kernel(kernel)

        self.assertTrue(np.array_equal(policy_ids, np.array([0, 1, 2])))
        self.assertEqual(sample_idxs, {0: [0, 2, 4, 6], 1: [1, 5], 2: [3, 7]})  # map policy id to agent list
        self.assertEqual(kernel, [0, 1, 0, 2, 0, 1, 0, 2])  # tiled into POOL_AGENTS

    def test_update_policies(self):
        policy_pool = self.nonrecurrent_policy_pool

        # Test with no policies in the policy store
        # All policies should be the learner policy
        policy_store = MockPolicyStore(0)
        policy_pool.update_policies(policy_ids=np.array([0, 1, 2]), store=policy_store)
        for pol in policy_pool.current_policies.values():
            self.assertEqual(pol['name'], 'learner')
            self.assertEqual(pol['policy'], policy_pool.learner_policy)

        # Sample 2 policies when there is only one policy in the policy store
        # Both policies should be Policy1
        policy_store = MockPolicyStore(1)
        policy_pool.update_policies(policy_ids=np.array([0, 1, 2]), store=policy_store)
        for pol in policy_pool.current_policies.values():
            self.assertEqual(pol['name'], 'Policy1')
            self.assertEqual(pol['policy'], policy_store.get_policy('Policy1'))

        # Sample 3 policies when there are 10 policies in the policy store
        # All sampled policies should be different
        policy_store = MockPolicyStore(10)
        policy_pool.update_policies(policy_ids=np.array([0, 1, 2, 3]), store=policy_store)
        self.assertEqual(len(set(p['name'] for p in policy_pool.current_policies.values())), 3)

        # Use all_selector
        policy_store = MockPolicyStore(5)
        policy_pool.update_policies(policy_ids=np.array([0, 1, 2, 3, 4, 5]), store=policy_store,
                                    policy_selector=pp.AllPolicySelector(seed=0))
        self.assertEqual(len(set(p['name'] for p in policy_pool.current_policies.values())), 5)

    def test_nonrecurrent_forward(self):
        policy_pool = self.nonrecurrent_policy_pool

        obs = torch.arange(POOL_AGENTS * OBS_DIM).view(POOL_AGENTS, OBS_DIM)
        atn, lgprob, val, _ = policy_pool.forwards(obs)

        for policy_id in policy_pool.policy_ids:
            samp = policy_pool.sample_idxs[policy_id]
            policy = policy_pool.learner_policy if policy_id == 0 \
                else policy_pool.current_policies[policy_id]['policy']
            atn1, lgprob1, _, val1 = policy(obs[samp])

            self.assertTrue(torch.equal(atn[samp], atn1))
            self.assertTrue(torch.equal(lgprob[samp], lgprob1))
            self.assertTrue(torch.equal(val[samp], val1))

    def test_update_scores(self):
        policy_pool = self.nonrecurrent_policy_pool
        # With the kernel [0, 1, 0, 2], agents 1 and 3 are learner, and agents 2 and 4 are different

        infos = [{1: {'return': 1}, 2: {'return': 2}, 3: {'return': 3}, 4: {'return': 4}},
                 {1: {'return': 10}, 2: {'return': 20}, 4: {'return': 40}}]
        pol1_name = policy_pool._get_policy_name(2)
        pol2_name = policy_pool._get_policy_name(4)

        policy_infos = policy_pool.update_scores(infos, 'return')
        self.assertEqual(policy_infos['learner'], [{'return': 1}, {'return': 3}, {'return': 10}])
        self.assertEqual(policy_infos[pol1_name], [{'return': 2}, {'return': 20}])
        self.assertEqual(policy_infos[pol2_name], [{'return': 4}, {'return': 40}])

        # policy_pool.scores only keep the last game's results
        self.assertEqual(policy_pool.scores['learner'], 10)
        self.assertEqual(policy_pool.scores[pol1_name], 20)
        self.assertEqual(policy_pool.scores[pol2_name], 40)

if __name__ == '__main__':
    unittest.main()
