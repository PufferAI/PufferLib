from pdb import set_trace as T
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

from torch import nn

import pufferlib.emulation
import pufferlib.exceptions
import pufferlib.models


class Policy(pufferlib.models.Policy):
    '''Based off of the DQN policy in MAgent'''
    def __init__(self, env, *args, input_size=256, hidden_size=256, output_size=256, **kwargs):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__(env)
        self.num_actions = self.action_space.n

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

    def encode_observations(self, observations):
        observations = observations.permute(0, 3, 1, 2)
        return self.network(observations), None

    def decode_actions(self, hidden, lookup):
        action = self.actor(hidden)
        value = self.value_function(hidden)
        return action, value

def make_battle_v4_env():
    '''MAgent Battle creation function'''
    try:
        from pettingzoo.magent import battle_v4 as battle
    except:
        raise pufferlib.exceptions.SetupError('magent', 'Battle V4')
    else:
        env = pufferlib.emulation.PettingZooPufferEnv(
            env_creator=aec_to_parallel_wrapper,
            env_args=[battle.env()],
        )
        return env
 
