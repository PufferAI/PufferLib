from pdb import set_trace as T

import pufferlib.pytorch
from pufferlib.environments.nethack import Policy


class Recurrent:
    input_size = 512
    hidden_size = 512
    num_layers = 1

