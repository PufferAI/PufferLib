import torch

import nmmo

import pufferlib
import pufferlib.binding
import pufferlib.emulation


class NeuralMMOBinding(pufferlib.binding.Base):
    def __init__(self):
        env_cls = pufferlib.emulation.PufferWrapper(nmmo.Env)
        self.original_observation_space =  nmmo.Env().observation_space(1)

        super().__init__('Neural MMO', env_cls)

        self.policy = Policy

    @property
    def custom_model_config(self):
        # These params are passed to the policy
        # per-agent observation/action spaces are computed for you
        # You may want to add in the original observation space, as below
        return {
            'input_size': 512,
            'hidden_size': 512,
            'lstm_layers': 1,
            'original_observation_space': self.original_observation_space,
            'observation_space': self.single_observation_space,
            'action_space': self.single_action_space,
        }

class Policy(pufferlib.binding.Policy):
    def __init__(self, original_observation_space, observation_space, action_space,
            input_size, hidden_size, lstm_layers):
        '''Simple custom PyTorch policy subclassing the pufferlib BasePolicy
        
        This requires only that you structure your network as an observation encoder,
        an action decoder, and a critic function. If you use our LSTM support, it will
        be added between the encoder and the decoder.
        '''
        super().__init__(input_size, hidden_size, lstm_layers)
        self.original_observation_space = original_observation_space
        self.observation_space = observation_space
        self.action_space = action_space

        # A dumb example encoder that applies a linear layer to agent self features
        observation_size = self.original_observation_space['Entity']['Continuous'].shape[1]
        self.encoder = torch.nn.Linear(observation_size, hidden_size)

        self.decoders = torch.nn.ModuleList([torch.nn.Linear(hidden_size, n)
                for n in action_space.nvec])
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def critic(self, hidden):
        return self.value_head(hidden)

    def encode_observations(self, env_outputs):
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            self.original_observation_space, env_outputs)
        env_outputs = env_outputs['Entity']['Continuous'][:, 0, :]
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions

