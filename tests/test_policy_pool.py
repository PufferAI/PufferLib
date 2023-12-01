# TODO: Update this test for the new policy pool code

import unittest
from unittest.mock import Mock

import torch

from pufferlib.policy_pool import PolicyPool


class MockPolicy:
    def get_action_and_value(self, obs, state, done):
        actions = torch.arange(4)
        logprobs = torch.arange(4, 8, dtype=torch.float32)
        values = torch.arange(8, 12, dtype=torch.float32)
        return actions, logprobs, None, values, state

class TestPolicyPool(unittest.TestCase):
    def setUp(self):
        self.mock_policy = MockPolicy()
        self.mock_policy.name = 'Policy1'

        self.policy_pool = PolicyPool(
            learner=self.mock_policy,
            learner_name=self.mock_policy.name,
            num_agents=2,
            num_envs=2,
        )

    def test_forwards(self):
        obs = torch.arange(12).view(4, 3)
        lstm_h = torch.arange(8).view(1, 4, 2)
        lstm_c = torch.arange(8, 16).view(1, 4, 2)
        lstm_state = (lstm_h, lstm_c)
        dones = torch.tensor([False, False, False, False])

        atn, lgprob, val, state = self.policy_pool.forwards(obs, lstm_state, dones)
        atn1, lgprob1, _, val1, state1 = self.mock_policy.get_action_and_value(
            None, lstm_state, None)

        self.assertTrue(torch.equal(atn, atn1))
        self.assertTrue(torch.equal(lgprob, lgprob1))
        self.assertTrue(torch.equal(val, val1))
        self.assertTrue(torch.equal(state[0], state1[0]))
        self.assertTrue(torch.equal(state[1], state1[1]))

    def test_update_scores(self):
        infos = [{'agent1': {'return': 5}, 'agent2': {'return': 10}},
                 {'agent1': {'return': 15}, 'agent2': {'return': 20}}]

        policy_infos = self.policy_pool.update_scores(infos, 'return')

        # Verify scores and num_scores attributes
        self.assertEqual(self.policy_pool.scores['Policy1'], [5, 10, 15, 20])

if __name__ == '__main__':
    unittest.main()
