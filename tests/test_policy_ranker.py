# TODO: Update for 0.5 rewrite of the ranker

import unittest
from unittest.mock import ANY, patch

from pufferlib.policy_ranker import OpenSkillRanker


class TestOpenSkillRanker(unittest.TestCase):

    def setUp(self):
        self.anchor = 'anchor'
        self.ranker = OpenSkillRanker(self.anchor)

    def test_add_policy(self):
        self.ranker.add_policy('test')
        self.assertIn('test', self.ranker.ratings())

    def test_update_ranks(self):
        scores = {'anchor': [10], 'policy2': [20]}
        self.ranker.update_ranks(scores)
        self.assertIn('anchor', self.ranker.ratings())
        self.assertIn('policy2', self.ranker.ratings())

    @patch('pickle.dump')
    def test_save_to_file(self, mock_pickle_dump):
        self.ranker.save_to_file('test.pkl')
        mock_pickle_dump.assert_called_once_with(self.ranker, ANY)

    @patch('pickle.load')
    def test_load_from_file(self, mock_pickle_load):
        mock_pickle_load.return_value = self.ranker
        instance = OpenSkillRanker.load_from_file('test.pkl')
        self.assertIsInstance(instance, OpenSkillRanker)
        mock_pickle_load.assert_called_once()

if __name__ == "__main__":
    unittest.main()
