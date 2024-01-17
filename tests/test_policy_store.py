# TODO: Update for 0.5
import copy

import unittest
from unittest.mock import patch

import torch

from pufferlib.models import Policy
from pufferlib.policy_store import DirectoryPolicyStore, FilePolicyRecord, MemoryPolicyStore, PolicyRecord, PolicySelector


class TestPolicyRecord(unittest.TestCase):
    def test_initialization(self):
        pr = PolicyRecord('policy1', torch.nn.Module())
        self.assertEqual(pr.name, 'policy1')
        self.assertIsInstance(pr.policy(), torch.nn.Module)

class TestPolicySelector(unittest.TestCase):
    def test_selection(self):
        ps = PolicySelector(2)
        policies = {
            'policy1': PolicyRecord('policy1', torch.nn.Module()),
            'policy2': PolicyRecord('policy2', torch.nn.Module()),
            'policy3': PolicyRecord('policy3', torch.nn.Module()),
        }
        selected_policies = ps.select_policies(policies)
        self.assertEqual(len(selected_policies), 2)

    def test_selection_with_exclusion(self):
        ps = PolicySelector(2, exclude_names={'policy2'})
        policies = {
            'policy1': PolicyRecord('policy1', torch.nn.Module()),
            'policy2': PolicyRecord('policy2', torch.nn.Module()),
            'policy3': PolicyRecord('policy3', torch.nn.Module()),
        }
        selected_policies = ps.select_policies(policies)
        self.assertEqual(len(selected_policies), 2)
        self.assertNotIn('policy2', [p.name for p in selected_policies])

class TestMemoryPolicyStore(unittest.TestCase):
    def setUp(self):
        self.store = MemoryPolicyStore()

    def test_add_policy(self):
        policy = torch.nn.Module()
        record = self.store.add_policy('policy1', policy)
        self.assertEqual(record.name, 'policy1')
        self.assertIsInstance(record.policy(), torch.nn.Module)

    def test_add_policy_error(self):
        policy = torch.nn.Module()
        self.store.add_policy('policy1', policy)
        with self.assertRaises(ValueError):
            self.store.add_policy('policy1', policy)

    def test_add_policy_copy(self):
        policy = torch.nn.Module()
        policy_copy = copy.deepcopy(policy)
        self.store.add_policy('policy1', policy)
        record_copy = self.store.add_policy_copy('policy2', 'policy1')
        self.assertNotEqual(id(record_copy.policy), id(policy))
        self.assertEqual(record_copy.name, 'policy2')

    def test_all_policies(self):
        policy = torch.nn.Module()
        self.store.add_policy('policy1', policy)
        self.store.add_policy('policy2', policy)
        all_policies = self.store._all_policies()
        self.assertEqual(len(all_policies), 2)

    def test_select_policies(self):
        selector = PolicySelector(1)
        policy = torch.nn.Module()
        self.store.add_policy('policy1', policy)
        selected_policies = self.store.select_policies(selector)
        self.assertEqual(len(selected_policies), 1)

class TestFilePolicyRecord(unittest.TestCase):
    def test_save(self):
        policy = torch.nn.Module()
        record = FilePolicyRecord('test', './test', policy)
        record.save()

    def test_load(self):
        policy = torch.nn.Module()
        record = FilePolicyRecord('test', './test')
        record.load()

    def test_policy_property(self):
        policy = torch.nn.Module()
        record = FilePolicyRecord('test', './test', policy)
        self.assertEqual(record.policy(), policy)

class TestDirectoryPolicyStore(unittest.TestCase):
    @patch('os.listdir', return_value=['test1.pt', 'test2.pt'])
    def test_all_policies(self, mock_listdir):
        store = DirectoryPolicyStore('./')
        policies = store._all_policies()
        self.assertEqual(len(policies), 2)
        self.assertIsInstance(policies['test1'], FilePolicyRecord)
        self.assertIsInstance(policies['test2'], FilePolicyRecord)

    @patch.object(FilePolicyRecord, 'save')
    def test_add_policy(self, mock_save):
        policy = torch.nn.Module()
        store = DirectoryPolicyStore('./')
        record = store.add_policy('test', policy)
        mock_save.assert_called_once()
        self.assertEqual(record.name, 'test')
        self.assertEqual(record.policy(), policy)

if __name__ == "__main__":
    unittest.main()
