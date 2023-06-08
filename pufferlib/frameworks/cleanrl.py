from pdb import set_trace as T

import torch
from torch.distributions import Categorical

import pufferlib
import pufferlib.models
import pufferlib.frameworks.base


def make_policy(policy_cls, recurrent_cls=torch.nn.LSTM,
        recurrent_args=[512, 128], recurrent_kwargs={'num_layers': 1}):
    '''Wrap a PyTorch model for use with CleanRL

    Args:
        policy_cls: A pufferlib.models.Policy subclass that implements the PufferLib model API
        recurrent_cls: Recurrent cell class to use. Defaults to torch.nn.LSTM.
        recurrent_args: Args to pass to recurrent_cls. Defaults to 512, 128 for LSTM.
        recurrent_kwargs: Kwargs to pass to recurrent_cls. Defaults to num_layers: 1 for LSTM. Set num_layers to 0 to disable the recurrent cell.

    Returns:
        A new PyTorch class wrapping your model that implements the CleanRL API
    '''
    assert issubclass(policy_cls, pufferlib.models.Policy)

    # Defaults for LSTM
    if recurrent_cls == torch.nn.LSTM:
        if len(recurrent_args) == 0:
            recurrent_args = [512, 128]
        if len(recurrent_kwargs) == 0:
            recurrent_args = {'num_layers': 1}

    is_recurrent = recurrent_kwargs['num_layers'] != 0
    if is_recurrent:
        policy_cls = pufferlib.frameworks.base.make_recurrent_policy(
            policy_cls, recurrent_cls, *recurrent_args, **recurrent_kwargs)

    class CleanRLPolicy(policy_cls):
        def _compute_hidden(self, x, lstm_state=None):
            if is_recurrent:
                batch_size = lstm_state[0].shape[1]
                x = x.reshape((-1, batch_size, x.shape[-1]))
                hidden, state, lookup = self.encode_observations(x, lstm_state)
                return hidden, state, lookup
            else:
                hidden, lookup = self.encode_observations(x)

            return hidden, lookup

        @property
        def lstm(self):
            return self.recurrent_policy

        # TODO: Cache value
        def get_value(self, x, lstm_state=None, done=None):
            if is_recurrent:
                hidden, lstm_state, lookup = self._compute_hidden(x, lstm_state)
            else:
                hidden, lookup = self._compute_hidden(x)
            return self.critic(hidden)

        # TODO: Compute seq_lens from done
        def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
            if is_recurrent:
                hidden, lstm_state, lookup = self._compute_hidden(x, lstm_state)
            else:
                hidden, lookup = self._compute_hidden(x)

            value = self.critic(hidden)
            flat_logits = self.decode_actions(hidden, lookup, concat=False)
            multi_categorical = [Categorical(logits=l) for l in flat_logits]

            if action is None:
                action = torch.stack([c.sample() for c in multi_categorical])
            else:
                action = action.view(-1, action.shape[-1]).T

            logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
            entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)

            if is_recurrent:
                return action.T, logprob, entropy, value, lstm_state
            else:
                return action.T, logprob, entropy, value

    return CleanRLPolicy
