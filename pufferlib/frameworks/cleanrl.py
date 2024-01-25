from pdb import set_trace as T

import torch
from torch.distributions import Categorical

import pufferlib.models


def sample_logits(logits, action=None):
    is_discrete = isinstance(logits, torch.Tensor)
    if is_discrete:
        logits = logits.unsqueeze(0) 

    multi_categorical = Categorical(logits=logits)

    batch = logits[0].shape[0]
    if action is None:
        action = multi_categorical.sample()
    else:
        action = action.view(batch, -1).T

    assert len(multi_categorical) == len(action)
    logprob = multi_categorical.log_prob(action).T.sum(1)
    entropy = multi_categorical.entropy().T.sum(1)

    if is_discrete:
        return action.squeeze(0), logprob.squeeze(0), entropy.squeeze(0)

    return action.T, logprob, entropy


class Policy(torch.nn.Module):
    '''Wrap a non-recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def get_value(self, x, state=None):
        _, value = self.policy(x)
        return value

    def get_action_and_value(self, x, action=None):
         logits, value = self.policy(x)
         action, logprob, entropy = sample_logits(logits, action)
         return action, logprob, entropy, value


class RecurrentPolicy(torch.nn.Module):
    '''Wrap a recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    @property
    def lstm(self):
        if hasattr(self.policy, 'recurrent'):
            return self.policy.recurrent
        elif hasattr(self.policy, 'lstm'):
            return self.policy.lstm
        else:
            raise ValueError('Policy must have a subnetwork named lstm or recurrent')

    def get_value(self, x, state=None):
        _, value, _ = self.policy(x, state)

    def get_action_and_value(self, x, state=None, action=None):
        logits, value, state = self.policy(x, state)
        action, logprob, entropy = sample_logits(logits, action)
        return action, logprob, entropy, value, state
