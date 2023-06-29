from pdb import set_trace as T

import torch
import torch.nn as nn

from pufferlib.pytorch import BatchFirstLSTM


def make_recurrent_policy(Policy, recurrent_cls=BatchFirstLSTM,
        *recurrent_args, **recurrent_kwargs):
    '''Wraps the given policy with an LSTM
    
    Called for you by framework-specific bindings
    '''
    class Recurrent(Policy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recurrent_policy = recurrent_cls(*recurrent_args, **recurrent_kwargs)

            # TODO: Generalize this
            self.input_size, self.hidden_size = recurrent_args

            # TODO: More general init options
            for name, param in self.recurrent_policy.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
 
        def encode_observations(self, x, state):
            # TODO: Check shapes
            #assert state is not None
            if state is not None:
                assert len(state) == 2

            assert len(x.shape) == 3

            B, TT, _ = x.shape
            x = x.reshape(B*TT, -1)

            hidden, lookup = super().encode_observations(x)
            assert hidden.shape == (B*TT, self.input_size)

            hidden = hidden.view(B, TT, self.input_size)
            hidden, state = self.recurrent_policy(hidden, state)
            hidden = hidden.reshape(B*TT, self.hidden_size)

            return hidden, state, lookup
    return Recurrent

