import pufferlib.models
from pufferlib.pytorch import LSTM


class Recurrent:
    input_size = 512
    hidden_size = 512
    num_layers = 1

class Policy(pufferlib.models.Convolutional):
    def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
            framestack=3, flat_size=64*5*6):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
            channels_last=True,
        )
