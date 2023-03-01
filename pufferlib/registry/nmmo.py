from pdb import set_trace as T

import torch

import nmmo

import pufferlib
import pufferlib.binding
import pufferlib.emulation

def create_binding():
    return pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        env_name='Neural MMO',
    )

class Policy(pufferlib.binding.Policy):
    def __init__(self, binding, input_size=512, hidden_size=512):
        '''Simple custom PyTorch policy subclassing the pufferlib BasePolicy
        
        This requires only that you structure your network as an observation encoder,
        an action decoder, and a critic function. If you use our LSTM support, it will
        be added between the encoder and the decoder.
        '''
        super().__init__(input_size, hidden_size)
        self.raw_single_observation_space = binding.raw_single_observation_space

        # A dumb example encoder that applies a linear layer to agent self features
        observation_size = binding.raw_single_observation_space['Entity'].shape[1]
        self.encoder = torch.nn.Linear(observation_size, hidden_size)

        self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden_size, n)
                for n in binding.single_action_space.nvec])
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            self.raw_single_observation_space, env_outputs)
        env_outputs = env_outputs['Entity'][:, 0, :]
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions

