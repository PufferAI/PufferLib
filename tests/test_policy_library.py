import unittest
import numpy as np

# Import your classes here:
# from your_module import PolicyLibrary, PolicyLibraryPool

class TestPolicyLibrary(unittest.TestCase):
    def setUp(self):
        self.policy_lib = PolicyLibrary()

    def test_add_policy(self):
        # Test adding new policy
        self.policy_lib.add_policy('Policy1', 'path1')
        self.assertEqual(self.policy_lib._model_paths['Policy1'], 'path1')

        # Test adding existing policy with overwrite_existing = False
        with self.assertRaises(ValueError):
            self.policy_lib.add_policy('Policy1', 'path2', overwrite_existing=False)

        # Test adding existing policy with overwrite_existing = True
        self.policy_lib.add_policy('Policy1', 'path2', overwrite_existing=True)
        self.assertEqual(self.policy_lib._model_paths['Policy1'], 'path2')


class TestPolicyLibraryPool(unittest.TestCase):
    def setUp(self):
        self.policy_lib = PolicyLibrary()
        self.policy_lib.add_policy('Policy1', 'path1')
        self.policy_lib.add_policy('Policy2', 'path2')
        self.policy_lib.add_policy('Policy3', 'path3')

        self.policy_pool = PolicyLibraryPool(self.policy_lib, batch_size=2, sample_weights=None)

        # Mock load_policy method since we can't test it without a real policy file
        self.policy_pool.load_policy = lambda policy_name: f"Loaded {policy_name}"

    def test_update_active_policies(self):
        # Test with no required policies
        self.policy_pool.update_active_policies()
        self.assertEqual(len(self.policy_pool._active_policies), 2)
        self.assertEqual(len(self.policy_pool._loaded_policies), 2)

        # Test with one required policy
        self.policy_pool.update_active_policies(required_policy_names=['Policy1'])
        self.assertEqual(len(self.policy_pool._active_policies), 2)
        self.assertEqual(len(self.policy_pool._loaded_policies), 2)
        self.assertIn('Loaded Policy1', self.policy_pool._active_policies)

if __name__ == '__main__':
    unittest.main()
