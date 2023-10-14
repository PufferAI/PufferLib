from pdb import set_trace as T

import pufferlib.pytorch
from pufferlib.registry.nethack import Policy


class Recurrent(pufferlib.pytorch.LSTM):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(input_size, hidden_size, num_layers)
