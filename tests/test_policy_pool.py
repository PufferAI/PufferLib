import unittest
from unittest.mock import Mock

import torch

from pufferlib.policy_pool import PolicyPool

class TestPolicyPool(unittest.TestCase):
    def setUp(self):
        self.mock_policy = Mock()
        self.mock_policy.model.get_action_and_value = Mock(return_value=(torch.tensor([1]), torch.tensor([2]), torch.tensor([3]), (torch.tensor([4]), torch.tensor([5]))))
        self.mock_policy.name = 'Policy1'

        self.policy_pool = PolicyPool(batch_size=10, sample_weights=[1, 2])
        self.policy_pool._policies = [self.mock_policy]

    def test_forwards(self):
        obs = torch.tensor([1, 2, 3])
        lstm_state = (torch.tensor([4]), torch.tensor([5]))
        dones = torch.tensor([False, False, False])

        all_actions, returns = self.policy_pool.forwards(obs, lstm_state, dones)

        self.assertEqual(all_actions.shape, (3, 1))  # Verify all_actions shape
        self.assertEqual(len(returns), 1)  # Verify returns length

        # Verify returned action, logprob, and value for each policy
        for atn, lgprob, val, lstm_state, samp in returns:
            self.assertTrue(torch.equal(atn, torch.tensor([1])))
            self.assertTrue(torch.equal(lgprob, torch.tensor([2])))
            self.assertTrue(torch.equal(val, torch.tensor([3])))
            self.assertTrue(torch.equal(lstm_state[0], torch.tensor([4])))
            self.assertTrue(torch.equal(lstm_state[1], torch.tensor([5])))

    def test_update_scores(self):
        infos = [{'agent1': {'return': 5}, 'agent2': {'return': 10}},
                 {'agent1': {'return': 15}, 'agent2': {'return': 20}}]

        policy_infos = self.policy_pool.update_scores(infos, 'return')

        # Verify policy_infos dictionary
        self.assertEqual(policy_infos['Policy1'], [{'return': 5}, {'return': 15}])

        # Verify scores and num_scores attributes
        self.assertEqual(self.policy_pool.scores['Policy1'], [5, 15])
        self.assertEqual(self.policy_pool.num_scores, 2)

if __name__ == '__main__':
    unittest.main()
