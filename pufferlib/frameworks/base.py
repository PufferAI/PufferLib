from pdb import set_trace as T

import torch


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, policy, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent = torch.nn.LSTM(
            input_size, hidden_size, num_layers=num_layers)

    def critic(self, hidden):
        return self.policy.critic(hidden)

    def encode_observations(self, x, state):
        # TODO: Generalize this
        batch_size = x.shape[0]
        x = x.reshape((-1, batch_size, x.shape[-1]))

        assert len(x.shape) == 3
        B, TT, _ = x.shape
        x = x.reshape(B*TT, -1)
        
        hidden, lookup = self.policy.encode_observations(x)
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.view(B, TT, self.input_size)
        hidden, state = self.recurrent(hidden, state)
        hidden = hidden.reshape(B*TT, self.hidden_size)

        return hidden, lookup, state

    def decode_actions(self, hidden, lookup, concat=None):
        return self.policy.decode_actions(hidden, lookup, concat=concat)