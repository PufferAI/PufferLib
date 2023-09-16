# Section 1: Emulation
import pufferlib.emulation

import nle, nmmo

def nmmo_creator():
    return pufferlib.emulation.PettingZooPufferEnv(env_creator=nmmo.Env)

def nethack_creator():
    return pufferlib.emulation.GymPufferEnv(env_creator=nle.env.NLE)

# Section 2: Vectorization
import pufferlib.vectorization

# vec = pufferlib.vectorization.Serial
vec = pufferlib.vectorization.Multiprocessing
# vec = pufferlib.vectorization.Ray

envs = vec(nmmo_creator, num_workers=2, envs_per_worker=2)

sync = True
if sync:
    obs = envs.reset()
else:
    envs.async_reset()
    obs, _, _, _ = envs.recv()

# Section 3: Policy
import torch
from torch import nn
import numpy as np

import pufferlib.frameworks.cleanrl

class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.encoder = nn.Linear(np.prod(
            envs.single_observation_space.shape), 128)
        self.decoders = nn.ModuleList([nn.Linear(128, n)
            for n in envs.single_action_space.nvec])
        self.value_head = nn.Linear(128, 1)

    def forward(self, env_outputs):
        env_outputs = env_outputs.reshape(env_outputs.shape[0], -1)
        hidden = self.encoder(env_outputs)
        actions = [dec(hidden) for dec in self.decoders]
        value = self.value_head(hidden)
        return actions, value

obs = torch.Tensor(obs)
policy = Policy(envs.driver_env)
cleanrl_policy = pufferlib.frameworks.cleanrl.Policy(policy)
actions = cleanrl_policy.get_action_and_value(obs)[0].numpy()
obs, rewards, dones, infos = envs.step(actions)
envs.close()

# Section 4: Registry Full Example
import torch

import pufferlib.models
import pufferlib.vectorization
import pufferlib.frameworks.cleanrl
import pufferlib.registry.nmmo

envs = pufferlib.vectorization.Multiprocessing(
    env_creator=pufferlib.registry.nmmo.make_env,
    num_workers=2, envs_per_worker=2)

policy = pufferlib.registry.nmmo.Policy(envs.driver_env)
cleanrl_policy = pufferlib.frameworks.cleanrl.Policy(policy)

obs = envs.reset()
obs = torch.Tensor(obs)
actions = cleanrl_policy.get_action_and_value(obs)[0].numpy()
obs, rewards, dones, infos = envs.step(actions)
envs.close()
