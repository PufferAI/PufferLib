from functools import partial
import torch

import pufferlib.models

class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy,
            input_size=128, hidden_size=128, num_layers=1):
        super().__init__(env, policy,
            input_size, hidden_size, num_layers)

class Policy(pufferlib.models.Convolutional):
    def __init__(self, env,
            input_size=128, hidden_size=128, output_size=128,
            framestack=3, flat_size=64*8*8):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
            channels_last=True,
        )
