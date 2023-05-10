from pdb import set_trace as T

import torch

import pufferlib
import pufferlib.models
import pufferlib.frameworks.cleanrl
import pufferlib.registry.classic_control
import pufferlib.vectorization.serial


# TODO: Integrate this test and others into a single cleanrl test file
# and add support for LSTM testing (i.e. default state)
def test_cleanrl_utils():
    binding = pufferlib.registry.classic_control.make_cartpole_binding()

    envs = pufferlib.vectorization.serial.VecEnv(binding, num_workers=2, envs_per_worker=2)
 
    obs = envs.reset()

    policy = pufferlib.frameworks.cleanrl.make_policy(
        pufferlib.models.Default, recurrent_kwargs={'num_layers': 0})(binding)

    obs = torch.tensor(obs).unsqueeze(1).float()
    actions = policy.get_action_and_value(obs)

if __name__ == '__main__':
    test_cleanrl_utils()