import copy
import unittest
from unittest.mock import MagicMock
from pufferlib.models import Policy

from pufferlib.policy_store import DirectoryPolicyStore, FilePolicyRecord, MemoryPolicyStore, PolicyRecord, PolicySelector
import unittest
from unittest.mock import MagicMock, patch

class MockPolicy:
    pass

class TestPolicyRecord(unittest.TestCase):
    def test_initialization(self):
        pr = PolicyRecord('policy1', MockPolicy())
        self.assertEqual(pr.name, 'policy1')
        self.assertIsInstance(pr.policy(), MockPolicy)
        self.assertEqual(pr.metadata, None)

class TestPolicySelector(unittest.TestCase):
    def test_selection(self):
        ps = PolicySelector(2)
        policies = {
            'policy1': PolicyRecord('policy1', MockPolicy()),
            'policy2': PolicyRecord('policy2', MockPolicy()),
            'policy3': PolicyRecord('policy3', MockPolicy()),
        }
        selected_policies = ps.select_policies(policies)
        self.assertEqual(len(selected_policies), 2)

    def test_selection_with_exclusion(self):
        ps = PolicySelector(2, exclude_names={'policy2'})
        policies = {
            'policy1': PolicyRecord('policy1', MockPolicy()),
            'policy2': PolicyRecord('policy2', MockPolicy()),
            'policy3': PolicyRecord('policy3', MockPolicy()),
        }
        selected_policies = ps.select_policies(policies)
        self.assertEqual(len(selected_policies), 2)
        self.assertNotIn('policy2', [p.name for p in selected_policies])

class TestMemoryPolicyStore(unittest.TestCase):
    def setUp(self):
        self.store = MemoryPolicyStore()

    def test_add_policy(self):
        policy = MockPolicy()
        record = self.store.add_policy('policy1', policy, {'key': 'value'})
        self.assertEqual(record.name, 'policy1')
        self.assertEqual(record.metadata, {'key': 'value'})
        self.assertIsInstance(record.policy(), MockPolicy)

    def test_add_policy_error(self):
        policy = MockPolicy()
        self.store.add_policy('policy1', policy)
        with self.assertRaises(ValueError):
            self.store.add_policy('policy1', policy)

    def test_add_policy_copy(self):
        policy = MockPolicy()
        policy_copy = copy.deepcopy(policy)
        self.store.add_policy('policy1', policy)
        record_copy = self.store.add_policy_copy('policy2', 'policy1')
        self.assertNotEqual(id(record_copy.policy), id(policy))
        self.assertEqual(record_copy.name, 'policy2')

    def test_all_policies(self):
        policy = MockPolicy()
        self.store.add_policy('policy1', policy)
        self.store.add_policy('policy2', policy)
        all_policies = self.store._all_policies()
        self.assertEqual(len(all_policies), 2)

    def test_select_policies(self):
        selector = PolicySelector(1)
        policy = MockPolicy()
        self.store.add_policy('policy1', policy)
        selected_policies = self.store.select_policies(selector)
        self.assertEqual(len(selected_policies), 1)

class TestFilePolicyRecord(unittest.TestCase):
    @patch('os.rename')
    @patch('torch.save')
    def test_save(self, mock_torch_save, mock_os_rename):
        policy = MagicMock(spec=Policy)
        record = FilePolicyRecord('test', './test.pt', policy)
        record.save()
        mock_torch_save.assert_called_once()
        mock_os_rename.assert_called_once()

    @patch('torch.load')
    @patch('os.path.exists', return_value=True)
    def test_load(self, mock_os_path_exists, mock_torch_load):
        policy = MagicMock(spec=Policy)
        policy = MagicMock(spec=Policy)
        create_policy_func = MagicMock(return_value=policy)
        record = FilePolicyRecord('test', './test.pt')
        record.load(create_policy_func)
        mock_torch_load.assert_called_once()
        self.assertIsNotNone(record.policy(create_policy_func))

    def test_policy_property(self):
        policy = MagicMock(spec=Policy)
        record = FilePolicyRecord('test', './test.pt', policy)
        self.assertEqual(record.policy(), policy)

class TestDirectoryPolicyStore(unittest.TestCase):
    @patch('os.listdir', return_value=['test1.pt', 'test2.pt'])
    def test_all_policies(self, mock_listdir):
        create_policy_func = MagicMock()
        store = DirectoryPolicyStore('./', create_policy_func)
        policies = store._all_policies()
        self.assertEqual(len(policies), 2)
        self.assertIsInstance(policies['test1'], FilePolicyRecord)
        self.assertIsInstance(policies['test2'], FilePolicyRecord)

    @patch.object(FilePolicyRecord, 'save')
    def test_add_policy(self, mock_save):
        policy = MagicMock(spec=Policy)
        create_policy_func = MagicMock()
        store = DirectoryPolicyStore('./', create_policy_func)
        record = store.add_policy('test', policy)
        mock_save.assert_called_once()
        self.assertEqual(record.name, 'test')
        self.assertEqual(record.policy(), policy)

if __name__ == "__main__":
    unittest.main()
