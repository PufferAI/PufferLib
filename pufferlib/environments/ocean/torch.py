from functools import partial
import torch

import pufferlib.models

RNN = partial(
    torch.nn.LSTM,
    input_size=128,
    hidden_size=128,
    num_layers=1
)

Policy = partial(
    pufferlib.models.Default,
    hidden_size=128
)
