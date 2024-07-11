from pdb import set_trace as T
from typing import List, Union

import torch
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

import pufferlib.models


# taken from torch.distributions.Categorical
def log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

# taken from torch.distributions.Categorical
def entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)

def sample_logits(logits: Union[torch.Tensor, List[torch.Tensor]],
        action=None, is_continuous=False):
    is_discrete = isinstance(logits, torch.Tensor)
    if is_continuous:
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)

        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    elif is_discrete:
        normalized_logits = [logits - logits.logsumexp(dim=-1, keepdim=True)]
        logits = [logits]
    else: # not sure what else it could be
        normalized_logits = [l - l.logsumexp(dim=-1, keepdim=True) for l in logits]


    if action is None:
        action = torch.stack([torch.multinomial(logits_to_probs(l), 1).squeeze() for l in logits])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    assert len(logits) == len(action)
    logprob = torch.stack([log_prob(l, a) for l, a in zip(normalized_logits, action)]).T.sum(1)
    logits_entropy = torch.stack([entropy(l) for l in normalized_logits]).T.sum(1)

    if is_discrete:
        return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)

    return action.T, logprob, logits_entropy


class Policy(torch.nn.Module):
    '''Wrap a non-recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.is_continuous = hasattr(policy, 'is_continuous') and policy.is_continuous

    def get_value(self, x, state=None):
        _, value = self.policy(x)
        return value

    def get_action_and_value(self, x, action=None):
         logits, value = self.policy(x)
         action, logprob, entropy = sample_logits(logits, action, self.is_continuous)
         return action, logprob, entropy, value

    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)


class RecurrentPolicy(torch.nn.Module):
    '''Wrap a recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.is_continuous = hasattr(policy.policy, 'is_continuous') and policy.policy.is_continuous

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
        action, logprob, entropy = sample_logits(logits, action, self.is_continuous)
        return action, logprob, entropy, value, state

    def forward(self, x, state=None, action=None):
        return self.get_action_and_value(x, state, action)
