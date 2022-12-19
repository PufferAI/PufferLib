from pdb import set_trace as T

import torch

import pufferlib


# TODO: Integrate this test and others into a single cleanrl test file
# and add support for LSTM testing (i.e. default state)
def test_cleanrl_utils():
    binding = pufferlib.bindings.registry.make_classic_control_bindings()

    envs = pufferlib.vecenvs.VecEnvs(binding, num_workers=2, envs_per_worker=2)
 
    obs = envs.reset()

    policy = binding.policy
    policy = pufferlib.cleanrl.make_cleanrl_policy(policy, lstm_layers=0)
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