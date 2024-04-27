from pdb import set_trace as T
import numpy as np
import pickle
import os

import torch
from torch import nn

from pufferlib.frameworks import cleanrl

numpy_to_torch_dtype_dict = {
    np.dtype('float32') : torch.float32,
    np.dtype('uint8') : torch.uint8,
    np.dtype('int16') : torch.int16,
    np.dtype('int32') : torch.int32,
    np.dtype('int64') : torch.int64,
    np.dtype('int8') : torch.int8,
}

def nativize_observation(observation, emulated):
    return nativize_tensor(
        observation,
        emulated.observation_space_dtype,
        emulated.emulated_observation_dtype,
    )
 
def nativize_tensor(sample, sample_space_dtype, emulated_dtype):
    '''Pytorch function that traverses obs_dtype and returns a structured
    object (dicts, lists, etc) with subviews into the observation tensor'''
    torch_dtype = numpy_to_torch_dtype_dict[sample_space_dtype]
    sample = sample.to(torch_dtype, copy=False)
    structured_view = _nativize_tensor(sample, sample_space_dtype, emulated_dtype)
    return structured_view

def _nativize_tensor(sample, sample_space_dtype, emulated_dtype, offset=0):
    if emulated_dtype.fields is None:
        dtype, shape = emulated_dtype.subdtype
        delta = np.prod(shape) * dtype.itemsize // sample_space_dtype.itemsize
        slice = sample.narrow(1, offset, delta)

        torch_dtype = numpy_to_torch_dtype_dict[dtype]
        # Inference should always be contiguous. Seems that LSTM dimenson reshape
        # breaks this. This is a workaround.
        slice = slice.view(torch_dtype).contiguous()
        slice = slice.view(sample.shape[0], *shape).to(torch_dtype, copy=False)
        return slice
    else:
        subviews = {}
        for name, (dtype, offset) in emulated_dtype.fields.items():
            subviews[name] = _nativize_tensor(sample, sample_space_dtype, dtype, offset)
        return subviews

class BatchFirstLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_first=True, **kwargs)

    def forward(self, input, hx):
        '''
        input: B x T x H
        h&c: B x T x H
        '''
        h, c       = hx
        h          = h.transpose(0, 1)
        c          = c.transpose(0, 1)
        hidden, hx = super().forward(input, [h, c])
        h, c       = hx
        h          = h.transpose(0, 1)
        c          = c.transpose(0, 1)
        return hidden, [h, c]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''CleanRL's default layer initialization'''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LSTM(nn.LSTM):
    def __init__(self, input_size=128, hidden_size=128, num_layers=1):
        super().__init__(input_size, hidden_size, num_layers)
        layer_init(self)
