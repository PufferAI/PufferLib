from functools import partial
import pufferlib.models

Recurrent = partial(
    pufferlib.models.LSTMWrapper,
    input_size=128,
    hidden_size=128,
    num_layers=1
)

Policy = partial(
    pufferlib.models.Default,
    hidden_size=128
)
