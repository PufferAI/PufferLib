import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from pufferlib.pytorch import numpy_to_torch_dtype_dict

LITTLE_BYTE_ORDER = sys.byteorder == "little"

# dtype of the tensor
# shape of the tensor
# starting element of the observation
# number of elements of the observation to take
NativeDTypeValue = Tuple[torch.dtype, List[int], int, int]
NativeDType = Union[NativeDTypeValue, Dict[str, Union[NativeDTypeValue, "NativeDType"]]]

def nativize_dtype(emulated) -> NativeDType:
    sample_dtype: np.dtype = emulated['observation_dtype']
    structured_dtype: np.dtype = emulated['emulated_observation_dtype']
    subviews, dtype, shape, offset, delta = _nativize_dtype(sample_dtype, structured_dtype)
    if subviews is None:
        return (dtype, shape, offset, delta)
    else:
        return subviews

def round_to(x, base):
    return int(base * np.ceil(x/base))

def _nativize_dtype(sample_dtype, structured_dtype, offset=0):
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


def nativize_tensor(observation, native_dtype):
    return _nativize_tensor(observation, native_dtype)


def compilable_cast(u8, dtype):
    if dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        n = dtype.itemsize
        bytes = [u8[..., i::n].to(dtype) for i in range(n)]
        if not LITTLE_BYTE_ORDER:
            bytes = bytes[::-1]

        bytes = sum(bytes[i] << (i * 8) for i in range(n))
        return bytes.view(dtype)
    return u8.view(dtype)


def _nativize_tensor(observation, native_dtype):
    if isinstance(native_dtype, tuple):
        dtype, shape, offset, delta = native_dtype
        torch._check_is_size(offset)
        torch._check_is_size(delta)
        slice = observation.narrow(1, offset, delta)
        slice = slice.view(dtype)
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
        emulated['observation_dtype'],
        emulated['emulated_observation_dtype'],
    )

def flattened_tensor_size(native_dtype):
    return _flattened_tensor_size(native_dtype)

def _flattened_tensor_size(native_dtype):
    if isinstance(native_dtype, tuple):
        return np.prod(native_dtype[1])
    else:
        res = 0
        for _, dtype in native_dtype.items():
            res += _flattened_tensor_size(dtype)
        return res
