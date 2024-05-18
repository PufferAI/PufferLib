import math
from pdb import set_trace as T
import numpy as np
import pickle
import os
import sys

import torch
from torch import nn

from pufferlib.frameworks import cleanrl

numpy_to_torch_dtype_dict = {
    np.dtype("float32"): torch.float32,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("int8"): torch.int8,
    np.dtype("uint8"): torch.uint8,
}


LITTLE_BYTE_ORDER = sys.byteorder == "little"

def nativize_dtype(sample_dtype: np.dtype, structured_dtype: np.dtype):
    return _nativize_dtype(sample_dtype, structured_dtype)


def _nativize_dtype(
    sample_dtype: np.dtype, structured_dtype: np.dtype, offset: int = 0
):
    if structured_dtype.fields is None:
        dtype, shape = structured_dtype.subdtype
        delta = int(np.prod(shape) * dtype.itemsize // sample_dtype.itemsize)
        return numpy_to_torch_dtype_dict[dtype], shape, delta, delta+offset
    else:
        subviews = {}
        for name, (dtype, _) in structured_dtype.fields.items():
            align = dtype.alignment
            offset = int((align * np.ceil(offset / align)).astype(np.int32))
            torch_dtype, shape, delta, new_offset = _nativize_dtype(
                sample_dtype, dtype, offset)
            subviews[name] = (torch_dtype, shape, delta, offset)
            offset = new_offset
        return subviews


def nativize_tensor(
    observation: torch.Tensor,
    native_dtype: tuple[torch.dtype, tuple[int], int, int]
    | dict[str, tuple[torch.dtype, tuple[int], int, int]],
):
    return _nativize_tensor(observation, native_dtype)


# torch.view(dtype) does not compile
# This is a workaround hack
def compilable_cast(u8, dtype): 
    n = dtype.itemsize
    bytes = [u8[..., i::n].to(dtype) for i in range(n)]
    if not LITTLE_BYTE_ORDER:
        bytes = bytes[::-1]

    bytes = sum(bytes[i] << (i * 8) for i in range(n))
    return bytes.view(dtype)

def _nativize_tensor(
    observation: torch.Tensor,
    native_dtype: tuple[torch.dtype, tuple[int], int, int]
    | dict[str, tuple[torch.dtype, tuple[int], int, int]],
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(native_dtype, tuple):
        dtype, shape, delta, offset = native_dtype
        torch._check_is_size(offset)
        torch._check_is_size(delta)
        # torch._check(observation.size(1) >= offset + delta)
        slice = observation.narrow(1, offset, delta)
        # slice = slice.contiguous()
        slice = compilable_cast(slice, dtype)
        #slice = slice.view(dtype)
        slice = slice.view(observation.shape[0], *shape)
        return slice
    else:
        subviews = {}
        for name, dtype in native_dtype.items():
            subviews[name] = _nativize_tensor(observation, dtype)
        return subviews


def nativize_observation(observation, emulated):
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
