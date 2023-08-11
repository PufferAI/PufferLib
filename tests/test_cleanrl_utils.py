from pdb import set_trace as T

import torch

import pufferlib
import pufferlib.models
import pufferlib.frameworks.cleanrl
import pufferlib.registry.classic_control
import pufferlib.vectorization


def test_cleanrl_utils():
    envs = pufferlib.vectorization.Serial(
        env_creator=pufferlib.registry.classic_control.make_cartpole_env,
        num_workers=2, envs_per_worker=2
    )
 
    obs = envs.reset()

    policy = pufferlib.models.Default(envs)
    policy = pufferlib.models.RecurrentWrapper(envs, policy)
    policy = pufferlib.frameworks.cleanrl.Policy(policy)

    obs = torch.tensor(obs).unsqueeze(1).float()
    actions = policy.get_action_and_value(obs)

if __name__ == '__main__':
    test_cleanrl_utils()