from pdb import set_trace as T

import pufferlib.pytorch
from pufferlib.environments.nethack import Policy

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)
