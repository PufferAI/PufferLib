from pdb import set_trace as T

import torch

import nle

import pufferlib


binding = pufferlib.bindings.auto(env_cls=nle.env.NLE)

envs = pufferlib.cleanrl.VecEnvs(binding.env_cls, binding.env_args, num_envs=2)

obs = envs.reset()

policy = binding.policy
policy = pufferlib.cleanrl.make_cleanrl_policy(policy, lstm_layers=1)
policy = policy(
    binding.observation_space,
    binding.action_space,
    32,
    32,
    1
)

obs = torch.tensor(obs).unsqueeze(1).float()
actions = policy.get_action_and_value(obs)

pass

