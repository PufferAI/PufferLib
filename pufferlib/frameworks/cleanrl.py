from pdb import set_trace as T

import torch
from torch.distributions import Categorical

import pufferlib
import pufferlib.models
import pufferlib.frameworks.base

class Policy(torch.nn.Module):
    '''Wrap a PyTorch model for use with CleanRL

    Args:
        policy_cls: A pufferlib.models.Policy subclass that implements the PufferLib model API
        recurrent_cls: Recurrent cell class to use. Defaults to torch.nn.LSTM.
        recurrent_args: Args to pass to recurrent_cls. Defaults to 512, 128 for LSTM.
        recurrent_kwargs: Kwargs to pass to recurrent_cls. Defaults to num_layers: 1 for LSTM. Set num_layers to 0 to disable the recurrent cell.

    Returns:
        A new PyTorch class wrapping your model that implements the CleanRL API
    '''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    @property
    def lstm(self):
        return self.policy.recurrent

    def _compute_hidden(self, x, state=None):
        hidden, lookup, state = self.policy.encode_observations(x, state)
        return hidden, lookup, state

    def get_value(self, x, state, done=None):
        hidden, lookup, _ = self._compute_hidden(x, state)
        return self.policy.critic(hidden)

    def get_action_and_value(self, x, state=None, done=None, action=None):
        hidden, lookup, state = self._compute_hidden(x, state)
        value = self.policy.critic(hidden)
        flat_logits = self.policy.decode_actions(hidden, lookup, concat=False)
        multi_categorical = [Categorical(logits=l) for l in flat_logits]

        if action is None:
            action = torch.stack([c.sample() for c in multi_categorical])
        else:
            action = action.view(-1, action.shape[-1]).T

        logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
        entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)

        if self.is_recurrent:
            return action.T, logprob, entropy, value, state
        return action.T, logprob, entropy, value