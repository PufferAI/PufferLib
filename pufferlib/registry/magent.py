from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

import torch
from torch import nn

import pufferlib.emulation
import pufferlib.models
import pufferlib.utils


class MAgentPostprocessor(pufferlib.emulation.Postprocessor):
    def features(self, obs, step):
        assert len(obs) == 1
        return list(obs.values())[0].transpose(2, 0, 1)

class Policy(pufferlib.models.Policy):
    '''Based off of the DQN policy in MAgent'''
    def __init__(self, binding, *args, input_size=256, hidden_size=256, output_size=256, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__(binding)
        self.observation_space = binding.single_observation_space
        self.num_actions = binding.raw_single_action_space.n

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(5, 32, 3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 32, 3)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(32*9*9, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(output_size, self.num_actions), std=0.01)
        self.value_function = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def critic(self, hidden):
        return self.value_function(hidden)

    def encode_observations(self, flat_observations):
        # TODO: Add flat obs support to emulation
        batch = flat_observations.shape[0]
        observations = flat_observations.reshape((batch,) + self.observation_space.shape)
        return self.network(observations), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        if concat:
            return action
        return [action]

def make_battle_v4_binding():
    '''MAgent Battle binding creation function'''
    try:
        from pettingzoo.magent import battle_v4 as battle
    except:
        raise pufferlib.utils.SetupError('MAgent (pettingzoo)')
    else:
        return pufferlib.emulation.Binding(
            env_cls=aec_to_parallel_wrapper,
            default_args=[battle.env()],
            env_name='MAgent Battle v4',
            postprocessor_cls=MAgentPostprocessor,
        )