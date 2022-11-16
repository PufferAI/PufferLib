import torch

from pufferlib.torch import BatchFirstLSTM

class BasePolicy(torch.nn.Module): #, ABC
    def __init__(self, input_size, hidden_size, lstm_layers=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

    #@abstractmethod
    def critic(self, hidden):
        #raise NotImplementedError
        return hidden

    def encode_observations(self, flat_observations):
        return hidden, lookup

    def decode_actions(self, flat_hidden, lookup):
        return logits

def make_recurrent_policy(Policy):
    class Recurrent(Policy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.lstm_layers > 0
            self.lstm = BatchFirstLSTM(
                self.input_size,
                self.hidden_size,
                self.lstm_layers,
            )

        def encode_observations(self, x, state, seq_lens):
            B, TT, _ = x.shape
            x = x.reshape(B*TT, -1)

            hidden, lookup = super().encode_observations(x)
            assert hidden.shape == (B*TT, self.hidden_size)

            hidden = hidden.view(B, TT, self.hidden_size)
            hidden, state = self.lstm(hidden, state)
            hidden = hidden.reshape(B*TT, self.hidden_size)

            return hidden, state, lookup
    return Recurrent

