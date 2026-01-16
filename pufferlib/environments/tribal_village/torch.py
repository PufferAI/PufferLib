"""
Ultra-minimal PyTorch policy for Tribal Village - optimized for maximum SPS.
"""

import torch
import torch.nn as nn
import pufferlib.pytorch
import pufferlib.models


class Policy(nn.Module):
    """Ultra-minimal policy optimized for speed over sophistication."""

    def __init__(self, env, hidden_size=64, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        # Dense observation processing - 21 layers -> hidden_size
        self.layer_proj = pufferlib.pytorch.layer_init(nn.Linear(21, hidden_size))

        # Action heads
        action_space = env.single_action_space
        self.actor = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
            for n in action_space.nvec
        ])

        # Value head
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, observations: torch.Tensor, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return (actions, value), hidden

    def forward_eval(self, observations: torch.Tensor, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Ultra-fast dense observation processing."""
        # observations shape: (batch, 21, 11, 11) - direct dense format from Nim

        # Global average pooling across spatial dimensions
        features = observations.float().mean(dim=(2, 3))  # (batch, 21) - one value per layer

        # Use the pre-initialized layer projection
        hidden = torch.relu(self.layer_proj(features))  # (batch, hidden_size)
        return hidden

    def decode_actions(self, hidden: torch.Tensor):
        """Simple action decoding."""
        logits = [head(hidden) for head in self.actor]
        values = self.value(hidden)
        return logits, values


class Recurrent(pufferlib.models.LSTMWrapper):
    """Minimal LSTM wrapper."""

    def __init__(self, env, policy, input_size=64, hidden_size=64):
        super().__init__(env, policy, input_size, hidden_size)