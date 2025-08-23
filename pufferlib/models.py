from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import inspect
import os

import pufferlib.emulation
import pufferlib.spaces
import pufferlib.pytorch

class Default(nn.Module):
    '''Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    '''
    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space,
                pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space,
                pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict) 
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict) 

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            num_obs = np.prod(env.single_observation_space.shape)
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
                nn.GELU(),
            )
            
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

        self.value = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        '''Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers).'''
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else: 
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        '''Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers).'''
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values

class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128, **kwargs):
        '''Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if 'layer_norm' in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        #self.pre_layernorm = nn.LayerNorm(hidden_size)
        #self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward_eval(self, observations, state):
        '''Forward function for inference. 3x faster than using LSTM directly'''
        hidden = self.policy.encode_observations(observations, state=state)
        h = state['lstm_h']
        c = state['lstm_c']

        # TODO: Don't break compile
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], 'LSTM state must be (h, c)'
            lstm_state = (h, c)
        else:
            lstm_state = None

        #hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        #hidden = self.post_layernorm(hidden)
        state['hidden'] = hidden
        state['lstm_h'] = hidden
        state['lstm_c'] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        '''Forward function for training with sequence processing'''
        if state is None:
            state = {}
        
        x = observations
        x_shape = x.shape
        obs_shape = self.obs_shape
        
        if len(x_shape) == len(obs_shape) + 1:
            B = x_shape[0]
            T = 1
            x_flat = x
        elif len(x_shape) == len(obs_shape) + 2:
            B, T = x_shape[:2]
            x_flat = x.reshape(B * T, *obs_shape)
        else:
            raise ValueError(f'Invalid input shape {x.shape}, expected {len(obs_shape)+1}D or {len(obs_shape)+2}D')
        
        hidden = self.policy.encode_observations(x_flat, state=state)
        if T > 1:
            hidden = hidden.reshape(B, T, self.input_size)
        
        h = state.get('lstm_h', None)
        c = state.get('lstm_c', None)
        
        if h is None or c is None:
            h = torch.zeros(1, B, self.hidden_size, device=hidden.device, dtype=hidden.dtype)
            c = torch.zeros(1, B, self.hidden_size, device=hidden.device, dtype=hidden.dtype)
        
        if T == 1:
            hidden = hidden.unsqueeze(1)  # [B, 1, input_size]
            
        lstm_out, (h_new, c_new) = self.lstm(hidden, (h, c))
        
        if T == 1:
            lstm_out = lstm_out.squeeze(1)  # [B, input_size]
        
        state['lstm_h'] = h_new.detach()
        state['lstm_c'] = c_new.detach()
        
        if T > 1:
            lstm_flat = lstm_out.reshape(B * T, self.hidden_size)
            logits, values = self.policy.decode_actions(lstm_flat)
            values = values.reshape(B, T)
        else:
            logits, values = self.policy.decode_actions(lstm_out)
        
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

    
def masked_logits(logits, mask, mask_value=-1e9):
    if mask is None:
        return logits
    mask_tensor = mask
    if not torch.is_tensor(mask_tensor):
        mask_tensor = torch.tensor(mask, device=logits.device)
    mask_bool = mask_tensor.bool()
    if mask_bool.dim() < logits.dim():
        for _ in range(logits.dim() - mask_bool.dim()):
            mask_bool = mask_bool.unsqueeze(-1)
    mask_bool = mask_bool.to(device=logits.device)
    return logits.masked_fill(~mask_bool, float(mask_value))

class PolicyHead(nn.Module):
    """Policy head for different action spaces"""
    def __init__(self, hidden_size, action_space):
        super().__init__()
        self.action_space = action_space
        
        if hasattr(action_space, 'n'):
            self.head = nn.Linear(hidden_size, action_space.n)
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)
            
        elif hasattr(action_space, 'nvec'):
            self.nvec = action_space.nvec
            self.heads = nn.ModuleList([
                nn.Linear(hidden_size, n) for n in self.nvec
            ])
            for head in self.heads:
                nn.init.xavier_uniform_(head.weight, gain=1.0)
                nn.init.zeros_(head.bias)
                
        elif hasattr(action_space, 'shape'):
            self.mean_head = nn.Linear(hidden_size, action_space.shape[0])
            self.logstd = nn.Parameter(torch.zeros(action_space.shape[0]))
            nn.init.xavier_uniform_(self.mean_head.weight, gain=1.0)
            nn.init.zeros_(self.mean_head.bias)
    
    def forward(self, hidden, action_mask=None):
        if hasattr(self.action_space, 'n'):
            logits = self.head(hidden)
            if action_mask is not None:
                logits = masked_logits(logits, action_mask)
            dist = torch.distributions.Categorical(logits=logits)
            return dist, {}
            
        elif hasattr(self.action_space, 'nvec'):
            logits_list = [head(hidden) for head in self.heads]
            if action_mask is not None:
                if isinstance(action_mask, list):
                    logits_list = [masked_logits(logits, mask) 
                                 for logits, mask in zip(logits_list, action_mask)]
            dists = [torch.distributions.Categorical(logits=logits) for logits in logits_list]
            return dists, {}
            
        elif hasattr(self.action_space, 'shape'):
            mean = self.mean_head(hidden)
            std = torch.exp(self.logstd.expand_as(mean))
            dist = torch.distributions.Normal(mean, std)
            return dist, {}

class SimpleSSM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, state_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.A_param = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.B = nn.Linear(input_dim, state_dim, bias=True)
        self.C = nn.Linear(state_dim, hidden_dim, bias=True)
        self.D = nn.Linear(input_dim, hidden_dim, bias=False)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.B.weight, gain=0.5)
            nn.init.zeros_(self.B.bias)
            nn.init.xavier_uniform_(self.C.weight, gain=1.0)
            nn.init.zeros_(self.C.bias)
            nn.init.xavier_uniform_(self.D.weight, gain=0.5)

        self.log_dt = nn.Parameter(torch.ones(state_dim) * -1.0)

        self.input_norm = nn.LayerNorm(input_dim)
        self.state_norm = nn.LayerNorm(state_dim)

    def get_discrete_params(self):
        dt = torch.exp(self.log_dt).clamp(min=1e-4, max=1.0)
        A_continuous = -torch.exp(self.A_param) - 0.5
        A_discrete = torch.exp(A_continuous * dt)
        
        return A_discrete, dt

    def forward_single(self, x, state=None):
        B = x.size(0)
        x = self.input_norm(x)
        
        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        
        A_discrete, dt = self.get_discrete_params()
        
        input_gate = torch.sigmoid(self.B(x))
        Bu = input_gate * dt.unsqueeze(0)
        
        new_state = A_discrete.unsqueeze(0) * state + Bu
        new_state = self.state_norm(new_state)

        output = self.C(new_state) + self.D(x)
        
        return output, new_state

    def forward_sequence(self, x, state=None):
        B, T = x.shape[:2]
        x = self.input_norm(x)
        
        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        A_discrete, dt = self.get_discrete_params()
        x_flat = x.reshape(B * T, self.input_dim)
        input_gates = torch.sigmoid(self.B(x_flat)).reshape(B, T, self.state_dim)
        Bu_seq = input_gates * dt.unsqueeze(0).unsqueeze(0)  # [B, T, state_dim]
        states = self._associative_scan(A_discrete, Bu_seq, state)
        states = self.state_norm(states.reshape(B * T, self.state_dim)).reshape(B, T, self.state_dim)
        states_flat = states.reshape(B * T, self.state_dim)
        C_out = self.C(states_flat).reshape(B, T, self.hidden_dim)
        D_out = self.D(x_flat).reshape(B, T, self.hidden_dim)
        
        output = C_out + D_out
        final_state = states[:, -1]
        
        return output, final_state

    def _associative_scan(self, A, Bu_seq, initial_state):
        B, T, state_dim = Bu_seq.shape
        log_A = torch.log(A.clamp(min=1e-8)).unsqueeze(0).unsqueeze(0)
        states = torch.zeros(B, T, state_dim, device=Bu_seq.device, dtype=Bu_seq.dtype)
        chunk_size = min(T, 64)
        current_state = initial_state
        
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_length = end - start
            chunk_states = torch.zeros(B, chunk_length, state_dim, device=Bu_seq.device)
            for t in range(chunk_length):
                current_state = A.unsqueeze(0) * current_state + Bu_seq[:, start + t]
                chunk_states[:, t] = current_state
            
            states[:, start:end] = chunk_states
        
        return states

    def forward(self, x, state=None, return_all_states=False):
        if x.dim() == 2:
            return self.forward_single(x, state)
        elif x.dim() == 3:
            return self.forward_sequence(x, state)


class AdaptiveGatingNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_dim=64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        with torch.no_grad():
            self.network[-1].bias.data = torch.tensor([0.1, -0.1])

    def forward(self, x, temperature=1.0, entropy_regularization=0.05):
        logits = self.network(x)
        
        if self.training and entropy_regularization > 0:
            entropy_bonus = torch.randn_like(logits) * entropy_regularization
            logits = logits + entropy_bonus 
        weights = F.softmax(logits / temperature, dim=-1)
        gating_entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean() 
        return weights, logits


class HybridCore(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128, ssm_state_dim=None,
                layer_norm=True, residual=True, gating_dropout=0.1, **kwargs):
        super().__init__()

        self.obs_shape = env.single_observation_space.shape
        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ssm_state_dim = ssm_state_dim or hidden_size // 2
        self.is_continuous = getattr(policy, 'is_continuous', False)
        
        self.use_layer_norm = layer_norm
        self.use_residual = residual

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.ssm = SimpleSSM(self.input_size, self.hidden_size, self.ssm_state_dim)

        self.gate_network = AdaptiveGatingNetwork(self.input_size)

        policy_hidden_size = getattr(policy, 'hidden_size', self.hidden_size)
        if self.hidden_size != policy_hidden_size:
            self.lstm_proj = nn.Linear(self.hidden_size, policy_hidden_size)
            self.ssm_proj = nn.Linear(self.hidden_size, policy_hidden_size)
        else:
            self.lstm_proj = None
            self.ssm_proj = None

        if self.use_layer_norm:
            self.ln_lstm = nn.LayerNorm(self.hidden_size)
            self.ln_ssm = nn.LayerNorm(self.hidden_size)
            
        if self.use_residual and self.input_size != self.hidden_size:
            self.residual_proj = nn.Linear(self.input_size, self.hidden_size)
        else:
            self.residual_proj = None

        self.register_buffer('training_steps', torch.tensor(0.0))
        self.warmup_steps = 50000.0
        self.gating_dropout = nn.Dropout(gating_dropout)
        self.register_buffer('recent_value_losses', torch.zeros(100))
        self.register_buffer('loss_idx', torch.tensor(0))
        self.performance_bias_strength = 0.3
        self.gating_entropy_regularization = 0.05
        self.register_buffer('recent_episode_returns', torch.zeros(50))
        self.register_buffer('episode_idx', torch.tensor(0))
        self.difficulty_adaptation_strength = 0.2
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'ln_' in name or 'norm' in name:
                continue
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                else:
                    nn.init.xavier_uniform_(param, gain=1.0)

    def _get_temperature(self):
        if not self.training:
            return 1.0
        progress = float(self.training_steps) / 1e5
        progress = max(0.0, min(1.0, progress))
        return 2.0 - 1.0 * progress

    def update_performance_tracking(self, value_loss):
        if self.training:
            idx = int(self.loss_idx) % 100
            self.recent_value_losses[idx] = float(value_loss.detach())
            self.loss_idx += 1
    
    def update_episode_tracking(self, episode_return):
        if self.training:
            idx = int(self.episode_idx) % 50
            self.recent_episode_returns[idx] = float(episode_return)
            self.episode_idx += 1
    
    def _get_performance_bias(self):
        if int(self.loss_idx) < 10:
            return 0.0
        recent_loss = self.recent_value_losses[:min(100, int(self.loss_idx))].mean()
        performance_difficulty = torch.clamp(recent_loss * 10.0, 0.0, 1.0)
        return self.performance_bias_strength * performance_difficulty
    def _get_difficulty_bias(self):
        if int(self.episode_idx) < 10:
            return 0.0
        recent_episodes = min(50, int(self.episode_idx))
        recent_returns = self.recent_episode_returns[:recent_episodes]
        if recent_episodes >= 20:
            early_half = recent_returns[:recent_episodes//2].mean()
            later_half = recent_returns[recent_episodes//2:].mean()
            difficulty_trend = torch.clamp((early_half - later_half) / (early_half + 1e-6), 0.0, 1.0)
        else:
            difficulty_trend = torch.clamp(recent_returns.var() / (recent_returns.mean().abs() + 1e-6), 0.0, 1.0)
        return self.difficulty_adaptation_strength * difficulty_trend

    def _process_cores(self, enc, lstm_state, ssm_state, T=1):
        B = enc.shape[0] if T == 1 else enc.shape[0]

        if lstm_state[0] is None:
            h0 = torch.zeros(1, B, self.hidden_size, device=enc.device, dtype=enc.dtype)
            c0 = torch.zeros(1, B, self.hidden_size, device=enc.device, dtype=enc.dtype)
            lstm_state = (h0, c0) 
        if T == 1:
            lstm_input = enc.unsqueeze(1)
            lstm_out, new_lstm_state = self.lstm(lstm_input, lstm_state)
            lstm_hidden = lstm_out.squeeze(1)
        else:
            lstm_hidden, new_lstm_state = self.lstm(enc, lstm_state)
        ssm_hidden, new_ssm_state = self.ssm(enc, ssm_state) 
        return lstm_hidden, ssm_hidden, new_lstm_state, new_ssm_state

    def _apply_normalization_and_residual(self, lstm_h, ssm_h, enc):
        if self.use_layer_norm:
            lstm_h = self.ln_lstm(lstm_h)
            ssm_h = self.ln_ssm(ssm_h)
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(enc)
            else:
                residual = enc
            lstm_h = lstm_h + residual
            ssm_h = ssm_h + residual
        return lstm_h, ssm_h

    def _compute_gating_and_mix(self, enc, lstm_h, ssm_h, lstm_values, ssm_values):
        temperature = self._get_temperature()

        gating_input = self.gating_dropout(enc) if self.training else enc
        routing_weights, routing_logits = self.gate_network(
            gating_input, temperature, self.gating_entropy_regularization)
        
        if self.training:
            performance_bias = self._get_performance_bias()
            difficulty_bias = self._get_difficulty_bias()
            total_lstm_bias = performance_bias + difficulty_bias
            
            bias_adjustment = torch.tensor([total_lstm_bias, -total_lstm_bias], 
                                         device=routing_logits.device, dtype=routing_logits.dtype)
            routing_logits = routing_logits + bias_adjustment.unsqueeze(0)
            routing_weights = F.softmax(routing_logits / temperature, dim=-1)
        
        w_lstm = routing_weights[..., 0:1]
        w_ssm = routing_weights[..., 1:2]

        if isinstance(lstm_h, list):
            mixed_logits = []
            for l_h, s_h in zip(lstm_h, ssm_h):
                stacked = torch.stack([l_h, s_h], dim=0)
                weights = torch.stack([w_lstm.squeeze(-1), w_ssm.squeeze(-1)], dim=0)
                temp_scaled = stacked / temperature
                weighted = temp_scaled + torch.log(weights.unsqueeze(-1) + 1e-8)
                mixed = temperature * torch.logsumexp(weighted, dim=0)
                mixed_logits.append(mixed)
            
            if self.training:
                mixed_logits = [logits + torch.randn_like(logits) * 0.1 
                              for logits in mixed_logits]
        else:
            stacked = torch.stack([lstm_h, ssm_h], dim=0)
            weights = torch.stack([w_lstm.squeeze(-1), w_ssm.squeeze(-1)], dim=0)
            temp_scaled = stacked / temperature
            weighted = temp_scaled + torch.log(weights.unsqueeze(-1) + 1e-8)
            mixed_logits = temperature * torch.logsumexp(weighted, dim=0)
            
            if self.training:
                mixed_logits = mixed_logits + torch.randn_like(mixed_logits) * 0.1
            
        mixed_values = w_lstm.squeeze(-1) * lstm_values.squeeze(-1) + \
                    w_ssm.squeeze(-1) * ssm_values.squeeze(-1)
        
        return mixed_logits, mixed_values, routing_weights

    def forward_eval(self, observations, state=None):
        state = state or {}
        enc = self.policy.encode_observations(observations, state=state)
        B = enc.shape[0]

        lstm_state = state.get('lstm_state', (None, None))
        ssm_state = state.get('ssm_state', None)
        lstm_h, ssm_h, new_lstm_state, new_ssm_state = self._process_cores(
            enc, lstm_state, ssm_state, T=1)
        lstm_h, ssm_h = self._apply_normalization_and_residual(lstm_h, ssm_h, enc)

        if self.lstm_proj is not None:
            lstm_h = self.lstm_proj(lstm_h)
        if self.ssm_proj is not None:
            ssm_h = self.ssm_proj(ssm_h)

        lstm_logits, lstm_values = self.policy.decode_actions(lstm_h)
        ssm_logits, ssm_values = self.policy.decode_actions(ssm_h)

        mixed_logits, mixed_values, _ = self._compute_gating_and_mix(
            enc, lstm_logits, ssm_logits, lstm_values, ssm_values)

        state['lstm_state'] = (new_lstm_state[0].detach(), new_lstm_state[1].detach())
        state['ssm_state'] = new_ssm_state.detach() if isinstance(new_ssm_state, torch.Tensor) else new_ssm_state

        if self.training:
            self.training_steps += 1.0
        
        return mixed_logits, mixed_values

    def forward(self, observations, state=None):
        state = state or {}
        x = observations

        x_shape = x.shape
        obs_shape = self.obs_shape
        
        if len(x_shape) == len(obs_shape) + 1:
            B = x_shape[0]
            T = 1
            x_flat = x
        elif len(x_shape) == len(obs_shape) + 2:
            B, T = x_shape[:2]
            x_flat = x.reshape(B * T, *obs_shape)
        else:
            raise ValueError(f'Invalid input shape {x.shape}')

        enc = self.policy.encode_observations(x_flat, state=state)
        if T > 1:
            enc = enc.reshape(B, T, self.input_size)

        lstm_state = state.get('lstm_state', (None, None))
        ssm_state = state.get('ssm_state', None)

        lstm_h, ssm_h, new_lstm_state, new_ssm_state = self._process_cores(
            enc, lstm_state, ssm_state, T)

        if T > 1:
            lstm_h_flat = lstm_h.reshape(B * T, self.hidden_size)
            ssm_h_flat = ssm_h.reshape(B * T, self.hidden_size)
            enc_flat = enc.reshape(B * T, self.input_size)
        else:
            lstm_h_flat = lstm_h
            ssm_h_flat = ssm_h
            enc_flat = enc

        lstm_h_flat, ssm_h_flat = self._apply_normalization_and_residual(
            lstm_h_flat, ssm_h_flat, enc_flat)

        if self.lstm_proj is not None:
            lstm_h_flat = self.lstm_proj(lstm_h_flat)
        if self.ssm_proj is not None:
            ssm_h_flat = self.ssm_proj(ssm_h_flat)

        lstm_logits, lstm_values = self.policy.decode_actions(lstm_h_flat)
        ssm_logits, ssm_values = self.policy.decode_actions(ssm_h_flat)

        mixed_logits, mixed_values, routing_weights = self._compute_gating_and_mix(
            enc_flat, lstm_logits, ssm_logits, lstm_values, ssm_values)

        if T > 1:
            mixed_values = mixed_values.reshape(B, T)

        state['lstm_state'] = (new_lstm_state[0].detach(), new_lstm_state[1].detach())
        state['ssm_state'] = new_ssm_state.detach() if isinstance(new_ssm_state, torch.Tensor) else new_ssm_state

        if self.training:
            self.training_steps += 1.0
        
        return mixed_logits, mixed_values