from pdb import set_trace as T

import torch

import pufferlib
import pufferlib.frameworks.cleanrl
import pufferlib.registry
from pufferlib.vecenvs import VecEnvs


# TODO: Integrate this test and others into a single cleanrl test file
# and add support for LSTM testing (i.e. default state)
def test_cleanrl_utils():
    binding = pufferlib.registry.make_classic_control_bindings()

    envs = VecEnvs(binding, num_workers=2, envs_per_worker=2)
 
    obs = envs.reset()

    policy = binding.policy
    policy = pufferlib.frameworks.cleanrl.make_cleanrl_policy(policy, lstm_layers=0)
    policy = policy(
        envs.single_observation_space,
        envs.single_action_space,
        32,
        32,
        0
    )

    obs = torch.tensor(obs).unsqueeze(1).float()
    actions = policy.get_action_and_value(obs)

if __name__ == '__main__':
    test_cleanrl_utils()