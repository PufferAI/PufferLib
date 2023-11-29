from pdb import set_trace as T
import numpy as np
import pickle
import os

import torch
from torch import nn

from pufferlib.frameworks import cleanrl


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
