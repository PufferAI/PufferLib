from pdb import set_trace as T

import torch

import pufferlib
import pufferlib.emulation
import pufferlib.models


def make_binding():
    '''Neural MMO binding creation function'''
    try:
        import nmmo
    except:
        raise pufferlib.utils.SetupError('Neural MMO (nmmo)')
    else:
        return pufferlib.emulation.Binding(
            env_cls=nmmo.Env,
            env_name='Neural MMO',
        )


class Policy(pufferlib.models.Policy):
    def __init__(self, binding, input_size=128, hidden_size=128):
        '''Default Neural MMO policy
        
        This is a dummy placeholder used to speed up tests because of the size of the
        Neural MMO observation space. It is not a good policy and will not learn anything.'''
        super().__init__(binding, input_size, hidden_size)
        self.featurized_single_observation_space = binding.featurized_single_observation_space

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
            self.featurized_single_observation_space, env_outputs)
        env_outputs = env_outputs['Entity'][:, 0, :]
        return self.encoder(env_outputs), None

    def decode_actions(self, hidden, lookup, concat=True):
        actions = [dec(hidden) for dec in self.decoders]
        if concat:
            return torch.cat(actions, dim=-1)
        return actions

