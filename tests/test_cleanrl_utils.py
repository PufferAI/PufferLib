from pdb import set_trace as T
import numpy as np

import torch
from torch.distributions import Categorical

import gym

import pufferlib
import pufferlib.models
import pufferlib.frameworks.cleanrl
import pufferlib.environments.classic_control
import pufferlib.vectorization


def test_cleanrl_utils():
    envs = pufferlib.vectorization.Serial(
        env_creator=pufferlib.environments.classic_control.make_env,
        num_envs=4, envs_per_worker=2
    )
 
    obs, info, _, _ = envs.reset()

    policy = pufferlib.models.Default(envs)
    policy = pufferlib.models.RecurrentWrapper(envs, policy)
    policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)

    obs = torch.tensor(obs).unsqueeze(1).float()
    actions = policy.get_action_and_value(obs)

def shape_check(a1, l1, e1, a2, l2, e2):
    assert a1.shape == a2.shape
    assert l1.shape == l2.shape
    assert e1.shape == e2.shape

def test_sample_logits():
    batch = 8

    d = gym.spaces.Discrete(5)
    d_logits = torch.randn(batch, 5)
    d_action = torch.tensor([d.sample() for _ in range(batch)])

    nvec = [3, 7, 4]
    md = gym.spaces.MultiDiscrete(nvec)
    md_logits = [torch.rand(batch, n) for n in nvec]
    md_action = torch.tensor(np.array([md.sample() for _ in range(batch)]))

    a1, l1, e1 = pufferlib.frameworks.cleanrl.sample_logits(d_logits)
    a2, l2, e2 = correct_sample_logits(d_logits)
    shape_check(a1, l1, e1, a2, l2, e2)

    a1, l1, e1 = pufferlib.frameworks.cleanrl.sample_logits(d_logits, action=d_action)
    a2, l2, e2 = correct_sample_logits(d_logits, d_action)
    shape_check(a1, l1, e1, a2, l2, e2)

    a1, l1, e1 = pufferlib.frameworks.cleanrl.sample_logits(md_logits)
    a2, l2, e2 = pufferlib.frameworks.cleanrl.sample_logits(md_logits, action=md_action)
    shape_check(a1, l1, e1, a2, l2, e2)

def correct_sample_logits(logits, action=None):
    '''A bad but known correct implementation'''
    if isinstance(logits, torch.Tensor):
        categorical = Categorical(logits=logits)
        if action is None:
            action = categorical.sample()
        else:
            action = action.view(-1)

        logprob = categorical.log_prob(action)
        entropy = categorical.entropy()
        return action, logprob, entropy

    multi_categorical = [Categorical(logits=l) for l in logits]

    if action is None:
        action = torch.stack([c.sample() for c in multi_categorical])
    else:
        action = action.view(-1, action.shape[-1]).T

    logprob = torch.stack([c.log_prob(a) for c, a in zip(multi_categorical, action)]).T.sum(1)
    entropy = torch.stack([c.entropy() for c in multi_categorical]).T.sum(1)
    return action, logprob, entropy

 
if __name__ == '__main__':
    test_cleanrl_utils()
    #test_sample_logits()
