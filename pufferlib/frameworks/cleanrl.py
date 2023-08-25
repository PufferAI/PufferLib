from pdb import set_trace as T

import torch
from torch.distributions import Categorical

import pufferlib.models

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

        # No good way to check for recurrent models
        self.is_recurrent = (isinstance(policy, pufferlib.models.RecurrentWrapper)
            or hasattr(policy, 'lstm') or hasattr(policy, 'recurrent'))

    @property
    def lstm(self):
        return self.policy.recurrent

    def get_value(self, x, state=None, done=None):
        if self.is_recurrent:
            _, value, _ = self.policy(x, state)
        else:
            _, value = self.policy(x)
        return value

    # TODO: Arg ordering for CleanRL for conv/lstm/puffer
    # TODO: Why does categorical need device here but not in cleanrl?
    def get_action_and_value(self, x, action=None, state=None, done=None):
        if self.is_recurrent:
            logits, value, state = self.policy(x, state)
        else:
            logits, value = self.policy(x)

        # Check for single action space
        if isinstance(logits, torch.Tensor):
            categorical = Categorical(logits=logits)
            if action is None:
                action = categorical.sample()
            else:
                action = action.view(-1)

            logprob = categorical.log_prob(action)
            entropy = categorical.entropy()
            if self.is_recurrent:
                return action, logprob, entropy, value, state
            return action, logprob, entropy, value

        multi_categorical = [Categorical(logits=l) for l in logits]

        if action is None:
            action = torch.stack([c.sample() for c in multi_categorical])
        else:
            action = action.view(-1, action.shape[-1]).T

        logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
        entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)

        if self.is_recurrent:
            return action.T, logprob, entropy, value, state
        return action.T, logprob, entropy, value