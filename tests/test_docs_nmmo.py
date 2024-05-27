# Section 1: Emulation
import pufferlib.emulation
import pufferlib.wrappers

import nle, nmmo

def nmmo_creator():
    env = nmmo.Env()
    env = pufferlib.wrappers.PettingZooTruncatedWrapper(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)

def nethack_creator():
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=nle.env.NLE)

# Section 2: Vectorization
import pufferlib.vector
backend = pufferlib.vector.Serial #or Multiprocessing, Ray
envs = pufferlib.vector.make(nmmo_creator, backend=backend, num_envs=4)

# Synchronous API - reset/step
obs, infos = envs.reset()

# Asynchronous API - async_reset, send/recv
envs.async_reset()
obs, rewards, terminals, truncateds, infos, env_id, mask = envs.recv()

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
obs, rewards, terminals, truncateds, infos = envs.step(actions)
envs.close()

# Section 4: Registry Full Example
import torch

import pufferlib.models
import pufferlib.vector
import pufferlib.frameworks.cleanrl
import pufferlib.environments.nmmo

make_env = pufferlib.environments.nmmo.env_creator()
envs = pufferlib.vector.make(make_env, backend=backend, num_envs=4)

policy = pufferlib.environments.nmmo.Policy(envs.driver_env)
cleanrl_policy = pufferlib.frameworks.cleanrl.Policy(policy)

env_outputs = envs.reset()[0]
obs = torch.from_numpy(env_outputs)
actions = cleanrl_policy.get_action_and_value(obs)[0].numpy()
next_obs, rewards, terminals, truncateds, infos = envs.step(actions)
envs.close()

# Section 5: Unpacking Observations
dtype = pufferlib.pytorch.nativize_dtype(envs.driver_env.emulated)
env_outputs = pufferlib.pytorch.nativize_tensor(obs, dtype)
print('Packed tensor:', obs.shape)
print('Unpacked:', env_outputs.keys())
from pdb import set_trace as T; T()
pass
