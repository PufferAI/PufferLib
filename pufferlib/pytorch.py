import sys
from pdb import set_trace as T
from typing import Dict, List, Tuple, Union
import contextlib

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.utils import logits_to_probs

import pufferlib
import pufferlib.models


numpy_to_torch_dtype_dict = {
    np.dtype("float64"): torch.float64,
    np.dtype("float32"): torch.float32,
    np.dtype("float16"): torch.float16,
    np.dtype("uint64"): torch.uint64,
    np.dtype("uint32"): torch.uint32,
    np.dtype("uint16"): torch.uint16,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int64"): torch.int64,
    np.dtype("int32"): torch.int32,
    np.dtype("int16"): torch.int16,
    np.dtype("int8"): torch.int8,
}


LITTLE_BYTE_ORDER = sys.byteorder == "little"

# USER NOTE: You should not get any errors in nativize.
# This is a complicated piece of code that attempts to convert
# flat bytes to structured tensors without breaking torch.compile.
# If you hit any errors, please post on discord.gg/puffer
# One exception: make sure you didn't change the dtype of your data
# ie by doing torch.Tensor(data) instead of torch.from_numpy(data)

# dtype of the tensor
# shape of the tensor
# starting element of the observation
# number of elements of the observation to take
# could be a namedtuple or dataclass
NativeDTypeValue = Tuple[torch.dtype, List[int], int, int]
NativeDType = Union[NativeDTypeValue, Dict[str, Union[NativeDTypeValue, "NativeDType"]]]

# TODO: handle discrete obs
# Spend some time trying to break this fn with differnt obs
def nativize_dtype(emulated: pufferlib.namespace) -> NativeDType:
    # sample dtype - the dtype of what we obtain from the environment (usually bytes)
    sample_dtype: np.dtype = emulated.observation_dtype
    # structured dtype - the gym.Space converted numpy dtype

    # the observation represents (could be dict, tuple, box, etc.)
    structured_dtype: np.dtype = emulated.emulated_observation_dtype
    subviews, dtype, shape, offset, delta = _nativize_dtype(sample_dtype, structured_dtype)
    if subviews is None:
        return (dtype, shape, offset, delta)
    else:
        return subviews

def round_to(x, base):
    return int(base * np.ceil(x/base))

def _nativize_dtype(sample_dtype: np.dtype,
        structured_dtype: np.dtype,
        offset: int = 0) -> NativeDType:
    if structured_dtype.fields is None:
        if structured_dtype.subdtype is not None:
            dtype, shape = structured_dtype.subdtype
        else:
            dtype = structured_dtype
            shape = (1,)

        delta = int(np.prod(shape))
        if sample_dtype.base.itemsize == 1:
            offset = round_to(offset, dtype.alignment)
            delta *= dtype.itemsize
        else:
            assert dtype.itemsize == sample_dtype.base.itemsize

        return None, numpy_to_torch_dtype_dict[dtype], shape, offset, delta
    else:
        subviews = {}
        start_offset = offset
        all_delta = 0
        for name, (dtype, _) in structured_dtype.fields.items():
            views, dtype, shape, offset, delta = _nativize_dtype(
                sample_dtype, dtype, offset)

            if views is not None:
                subviews[name] = views
            else:
                subviews[name] = (dtype, shape, offset, delta)

            offset += delta
            all_delta += delta

        return subviews, dtype, shape, start_offset, all_delta


def nativize_tensor(
    observation: torch.Tensor,
    native_dtype: NativeDType,
) -> torch.Tensor | dict[str, torch.Tensor]:
    return _nativize_tensor(observation, native_dtype)


# torch.view(dtype) does not compile
# This is a workaround hack
# @thatguy - can you figure out a more robust way to handle cast?
# I think it may screw up for non-uint data... so I put a hard .view
# fallback that breaks compile
def compilable_cast(u8, dtype):
    if dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        n = dtype.itemsize
        bytes = [u8[..., i::n].to(dtype) for i in range(n)]
        if not LITTLE_BYTE_ORDER:
            bytes = bytes[::-1]

        bytes = sum(bytes[i] << (i * 8) for i in range(n))
        return bytes.view(dtype)
    return u8.view(dtype)  # breaking cast


def _nativize_tensor(
    observation: torch.Tensor, native_dtype: NativeDType
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(native_dtype, tuple):
        dtype, shape, offset, delta = native_dtype
        torch._check_is_size(offset)
        torch._check_is_size(delta)
        # Important, we are assuming that obervations of shape
        # [N, D] where N is number of examples and D is number of
        # bytes per example is being passed in
        slice = observation.narrow(1, offset, delta)
        # slice = slice.contiguous()
        # slice = compilable_cast(slice, dtype)
        slice = slice.view(dtype)
        slice = slice.view(observation.shape[0], *shape)
        return slice
    else:
        subviews = {}
        for name, dtype in native_dtype.items():
            subviews[name] = _nativize_tensor(observation, dtype)
        return subviews


def nativize_observation(observation, emulated):
    # TODO: Any way to check that user has not accidentally cast data to float?
    # float is natively supported, but only if that is the actual correct type
    return nativize_tensor(
        observation,
        emulated.observation_dtype,
        emulated.emulated_observation_dtype,
    )


def flattened_tensor_size(native_dtype: tuple[torch.dtype, tuple[int], int, int]):
    return _flattened_tensor_size(native_dtype)


def _flattened_tensor_size(
    native_dtype: tuple[torch.dtype, tuple[int], int, int],
) -> int:
    if isinstance(native_dtype, tuple):
        return np.prod(native_dtype[1])  # shape
    else:
        res = 0
        for _, dtype in native_dtype.items():
            res += _flattened_tensor_size(dtype)
        return res


class BatchFirstLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_first=True, **kwargs)

    def forward(self, input, hx):
        """
        input: B x T x H
        h&c: B x T x H
        """
        h, c = hx
        h = h.transpose(0, 1)
        c = c.transpose(0, 1)
        hidden, hx = super().forward(input, [h, c])
        h, c = hx
        h = h.transpose(0, 1)
        c = c.transpose(0, 1)
        return hidden, [h, c]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """CleanRL's default layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LSTM(nn.LSTM):
    def __init__(self, input_size=128, hidden_size=128, num_layers=1):
        super().__init__(input_size, hidden_size, num_layers)
        layer_init(self)

def cycle_selector(sample_idx, num_policies):
    return sample_idx % num_policies

class PolicyPool(torch.nn.Module):
    def __init__(self, vecenv, policies, learner_mask, device,
            policy_selector=cycle_selector):
        '''Experimental utility for running multiple different policies'''
        super().__init__()
        assert len(learner_mask) == len(policies)
        self.policy_map = torch.tensor([policy_selector(i, len(policies))
            for i in range(vecenv.num_agents)])
        self.learner_mask = learner_mask
        self.policies = torch.nn.ModuleList(policies)
        self.vecenv = vecenv

        # Assumes that all policies have the same LSTM or no LSTM
        self.lstm = policies[0].lstm if hasattr(policies[0], 'lstm') else None

        # Allocate buffers
        self.actions = torch.zeros(vecenv.num_agents,
            *vecenv.single_action_space.shape, dtype=int).to(device)
        self.logprobs = torch.zeros(vecenv.num_agents).to(device)
        self.entropy = torch.zeros(vecenv.num_agents).to(device)
        self.values = torch.zeros(vecenv.num_agents).to(device)

    def forward(self, obs, state=None, action=None):
        policy_map = self.policy_map[self.vecenv.batch_mask]
        for policy_idx in range(len(self.policies)):
            policy = self.policies[policy_idx]
            is_learner = self.learner_mask[policy_idx]
            samp_mask = policy_map == policy_idx

            context = torch.no_grad() if is_learner else contextlib.nullcontext()
            with context:
                ob = obs[samp_mask]
                if len(ob) == 0:
                    continue

                if state is not None:
                    lstm_h, lstm_c = state
                    h = lstm_h[:, samp_mask]
                    c = lstm_c[:, samp_mask]
                    atn, lgprob, entropy, val, (h, c) = policy(ob, (h, c))
                    lstm_h[:, samp_mask] = h
                    lstm_c[:, samp_mask] = c
                else:
                    atn, lgprob, _, val = policy(ob)

            self.actions[samp_mask] = atn
            self.logprobs[samp_mask] = lgprob
            self.entropy[samp_mask] = entropy
            self.values[samp_mask] = val.flatten()

        return self.actions, self.logprobs, self.entropy, self.values, (lstm_h, lstm_c)

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

def entropy_probs(logits, probs):
    p_log_p = logits * probs
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
        logits = logits.unsqueeze(0)
    else: #multi-discrete
        logits = torch.nn.utils.rnn.pad_sequence(
            [l.transpose(0,1) for l in logits], 
            batch_first=False, 
            padding_value=-torch.inf
        ).permute(1,2,0)

    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = logits_to_probs(normalized_logits)

    if action is None:
        action = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True)
        action = action.reshape(probs.shape[:-1])
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T
        probs = logits_to_probs(normalized_logits)

    assert len(logits) == len(action)
    logprob = log_prob(normalized_logits, action)
    logits_entropy = entropy(normalized_logits).sum(0)

    if is_discrete:
        return action.squeeze(0), logprob.squeeze(0), logits_entropy.squeeze(0)

    return action.T, logprob.sum(0), logits_entropy



