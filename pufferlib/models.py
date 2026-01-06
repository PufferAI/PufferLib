from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib.emulation
import pufferlib.spaces

# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class MinGRULayer(Module):
    def __init__(self, dim, expansion_factor=1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        self.proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        nn.init.orthogonal_(self.to_hidden_and_gate.weight)

        self.to_out = Linear(dim_inner, dim, bias = False)
        nn.init.orthogonal_(self.to_out.weight)

        self.norm = torch.nn.RMSNorm(dim)

    def forward(self, x, prev_hidden = None):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel
            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        if self.proj_out:
            out = self.to_out(out)

        out = out + x
        out = self.norm(out)

        return out, next_prev_hidden

class DefaultEncoder(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
        else:
            num_obs = np.prod(env.single_observation_space.shape)
            dtype = env.single_observation_space.dtype

        self.dtype = dtype
        self.encoder = pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size))

    def forward(self, observations):
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)

        hidden = self.encoder(observations.float())
        return F.gelu(hidden)

class DefaultDecoder(nn.Module):
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = pufferlib.pytorch.layer_init(
                    nn.Linear(hidden_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, num_atns), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01)
            self.decoder_logstd = nn.Parameter(torch.zeros(
                1, env.single_action_space.shape[0]))

        self.value_function = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)


    def forward(self, hidden):
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value_function(hidden)
        return logits, values
 

class Default(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.num_layers = num_layers
        self.obs_shape = env.single_observation_space.shape
        self.encoder = DefaultEncoder(env, hidden_size)
        self.decoder = DefaultDecoder(env, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]

            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')

            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()

            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, x, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        assert state[0].shape[1] == state[1].shape[1] == x.shape[0], 'LSTM state must be (h, c)'
        h = self.encoder(x)
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c

        logits, values = self.decoder(h)
        return logits, values, (lstm_h, lstm_c)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.encoder(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        h = h.transpose(0, 1)
        h, (lstm_h, lstm_c) = self.lstm.forward(h)
        h = h.transpose(0, 1)

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.decoder(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values

class GRU(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.num_layers = num_layers
        self.obs_shape = env.single_observation_space.shape
        self.encoder = DefaultEncoder(env, hidden_size)
        self.decoder = DefaultDecoder(env, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = torch.nn.RMSNorm(hidden_size)

        for i in range(num_layers):
            cell = self.cell[i]

            w_ih = getattr(self.gru, f'weight_ih_l{i}')
            w_hh = getattr(self.gru, f'weight_hh_l{i}')
            b_ih = getattr(self.gru, f'bias_ih_l{i}')
            b_hh = getattr(self.gru, f'bias_hh_l{i}')

            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()

            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h,)

    def forward_eval(self, x, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        assert state[0].shape[1] == x.shape[0]
        h = self.encoder(x)
        state = state[0]
        for i in range(self.num_layers):
            h_in = h    
            h = self.cell[i](h, state[i])
            state[i] = h
            h = h + h_in
            h = self.norm(h)

        logits, values = self.decoder(h)
        return logits, values, (state,)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.encoder(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        h = h.transpose(0, 1)
        h_in = h
        h, _ = self.gru.forward(h)
        h = h + h_in
        h = self.norm(h)
        h = h.transpose(0, 1)

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.decoder(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values


class MinGRU(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, expansion_factor=2, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.expansion_factor = expansion_factor
        self.obs_shape = env.single_observation_space.shape
        self.encoder = DefaultEncoder(env, hidden_size)
        self.decoder = DefaultDecoder(env, hidden_size)
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.mingru = nn.ModuleList([MinGRULayer(hidden_size, expansion_factor) for _ in range(num_layers)])

    def initial_state(self, batch_size, device):
        state = torch.zeros(self.num_layers, batch_size, self.hidden_size*self.expansion_factor, device=device)
        return (state,)

    def forward_eval(self, x, state):
        state = state[0]
        assert state.shape[1] == x.shape[0]
        h = self.encoder(x)
        h = h.unsqueeze(1)
        state = state.unsqueeze(2)
        state_out = []
        for i in range(self.num_layers):
            h, s = self.mingru[i](h, state[i])
            state_out.append(s)

        h = h.squeeze(1)
        state = torch.stack(state_out, 0).squeeze(2)
        logits, values = self.decoder(h)
        return logits, values, (state,)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.encoder(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        state = self.initial_state(B, h.device)[0].unsqueeze(2)
        for i in range(self.num_layers):
            h, _ = self.mingru[i](h, state[i])

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.decoder(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values

class Mamba(nn.Module):
    def __init__(self, env, hidden_size=128, num_layers=1, d_state=32, d_conv=4, expand=1):
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.obs_shape = env.single_observation_space.shape
        self.encoder = DefaultEncoder(env, hidden_size)
        self.decoder = DefaultDecoder(env, hidden_size)

        self.num_layers = num_layers
        from mamba_ssm import Mamba2
        self.mamba = nn.ModuleList([Mamba2(d_model=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)])

    def initial_state(self, batch_size, device):
        conv_state = torch.zeros(
            self.num_layers,
            batch_size,
            self.mamba[0].d_conv,
            self.mamba[0].conv1d.weight.shape[0],
            device=device,
            dtype=self.mamba[0].conv1d.weight.dtype,
        ).transpose(2, 3).to(device)
        ssm_state = torch.zeros(
            self.num_layers,
            batch_size,
            self.mamba[0].nheads,
            self.mamba[0].headdim,
            self.mamba[0].d_state,
            device=device,
            dtype=self.mamba[0].in_proj.weight.dtype,
        ).to(device)
        return conv_state, ssm_state

    def forward_eval(self, x, state):
        h = self.encoder(x)
        h = h.unsqueeze(1)
        conv_state, ssm_state = state
        for i in range(self.num_layers):
            h, conv_state[i], ssm_state[i] = self.mamba[i].step(h, conv_state[i], ssm_state[i])

        state = (conv_state, ssm_state)
        h = h.squeeze(1)
        logits, values = self.decoder(h)
        return logits, values, state

    def forward(self, x):
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.encoder(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        for i in range(self.num_layers):
            h = self.mamba[i](h)

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.decoder(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values

class LSTMWrapper(nn.Module):
    def __init__(self, env, make_policy_fn, hidden_size=128, num_layers=1, **kwargs):
        '''Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__()
        self.obs_shape = env.single_observation_space.shape
        input_size = hidden_size

        # NOTE: LSTM API is changing. Should revisit this.
        self.policy = make_policy_fn()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if 'layer_norm' in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]

            w_ih = getattr(self.lstm, f'weight_ih_l{i}')
            w_hh = getattr(self.lstm, f'weight_hh_l{i}')
            b_ih = getattr(self.lstm, f'bias_ih_l{i}')
            b_hh = getattr(self.lstm, f'bias_hh_l{i}')

            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()

            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, x, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        assert state[0].shape[1] == state[1].shape[1] == x.shape[0], 'LSTM state must be (h, c)'
        h = self.policy.encode_observations(x)
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c

        logits, values = self.policy.decode_actions(h)
        return logits, values, (lstm_h, lstm_c)

    def forward(self, x):
        '''Forward function for training. Uses LSTM for fast time-batching'''
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        assert x_shape[-space_n:] == space_shape, f'Invalid input tensor shape {x.shape} != {space_shape}'

        B, TT = x_shape[:2]
        x = x.reshape(B*TT, *space_shape)
        h = self.policy.encode_observations(x)
        assert h.shape == (B*TT, self.input_size)
        h = h.reshape(B, TT, self.input_size)

        h = h.transpose(0, 1)
        h, (lstm_h, lstm_c) = self.lstm.forward(h)
        h = h.transpose(0, 1)

        flat_hidden = h.reshape(B*TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        return logits, values

class Convolutional(nn.Module):
    def __init__(self, env, *args, framestack, flat_size,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=False, downsample=1, **kwargs):
        '''The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer
        
        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without.'''
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        #TODO: Remove these from required params
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network= nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(output_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]
        return self.network(observations.float() / 255.0)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value

class ProcgenResnet(nn.Module):
    '''Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper.'''
    def __init__(self, env, cnn_width=16, mlp_width=256):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2*cnn_width, 2*cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(
                nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden
 
    def decode_actions(self, hidden):
        '''linear decoder function'''
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)
