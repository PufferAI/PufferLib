from pdb import set_trace as T

import torch
from torch.distributions import Categorical

from pufferlib.frameworks import make_recurrent_policy, BasePolicy


def make_cleanrl_policy(policy_cls, lstm_layers=0):
    assert issubclass(policy_cls, BasePolicy)

    class CleanRLPolicy(policy_cls):
        '''Temporary hack to get framework running with CleanRL

        Their LSTMs are kind of weird. Need to figure this out'''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if lstm_layers > 0:
                self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, lstm_layers)

        # TODO: Cache value
        def get_value(self, x, lstm_state=None, done=None):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
                #lstm_state = [lstm_state[0].unsqueeze(1), lstm_state[1].unsqueeze(1)]

            hidden, _ = self.encode_observations(x)
            hidden, lstm_state = self._compute_lstm(hidden, lstm_state, done)

            return self.value_head(hidden)

        # TODO: Compute seq_lens from done, replace with PufferLib LSTM
        def _compute_lstm(self, hidden, lstm_state, done):
            batch_size = lstm_state[0].shape[1]
            hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
            done = done.reshape((-1, batch_size))
            new_hidden = []
            for h, d in zip(hidden, done):
                h, lstm_state = self.lstm(
                    h.unsqueeze(0),
                    (
                        (1.0 - d).view(1, -1, 1) * lstm_state[0],
                        (1.0 - d).view(1, -1, 1) * lstm_state[1],
                    ),
                )
                new_hidden += [h]
            return torch.flatten(torch.cat(new_hidden), 0, 1), lstm_state

        # TODO: Compute seq_lens from done
        def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            hidden, _ = self.encode_observations(x)
            hidden, lstm_state = self._compute_lstm(hidden, lstm_state, done)

            value = self.value_head(hidden)
            flat_logits = self.decode_actions(hidden, None, concat=False)

            multi_categorical = [Categorical(logits=l) for l in flat_logits]

            if action is None:
                action = torch.stack([c.sample() for c in multi_categorical])
            else:
                action = action.view(-1, action.shape[-1]).T

            logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
            entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)

            if lstm_layers > 0:
                return action.T, logprob, entropy, value, lstm_state
            return action.T, logprob, entropy, value
   
    return CleanRLPolicy