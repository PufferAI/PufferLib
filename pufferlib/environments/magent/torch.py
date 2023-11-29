from torch import nn

import pufferlib.models


class Policy(pufferlib.models.Policy):
    '''Based off of the DQN policy in MAgent'''
    def __init__(self, env, hidden_size=256, output_size=256, kernel_num=32):
        '''The CleanRL default Atari policy: a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword arguments. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__(env)
        self.num_actions = self.action_space.n

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(5, kernel_num, 3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(kernel_num, kernel_num, 3)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(kernel_num*9*9, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
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
