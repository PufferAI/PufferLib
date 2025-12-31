# This file defines PyTorch FakeTensor kernels. These operations
# match the input and output parameters and tensor shapes of the real
# operations defined in our C++ extensions and are required by PyTorch to
# add compile support. The PyTorch devs in their infinite wisdom have
# decided not to add compile support to the C++ API directly and to prioritize
# this interface over the C++ FakeTensor interface, so go complain to them
# if you think this is jank.

import torch

from pufferlib import _C


@torch.library.register_fake("_C::mingru_gate")
def mingru_gate_abstract(state: torch.Tensor, gate: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(state)

@torch.library.register_fake("_C::log_coeffs_and_values")
def log_coeffs_and_values_abstract(gate, hidden):
    log_coeffs = torch.empty_like(gate)
    values = torch.empty_like(hidden)
    return log_coeffs, values

@torch.library.register_fake("_C::policy_forward")
def policy_forward_abstract(obs, state):
    batch = obs.size(0)
    logits = torch.empty(batch, 3)
    value = torch.empty(batch, 1)
    state_out = torch.empty_like(state)
    return logits, value, state_out
