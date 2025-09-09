import einops
import torch
from torch import nn

import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)


class Policy(nn.Module):
    def __init__(self, env, cnn_channels=128, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.out_width = 11
        self.out_height = 11
        # Get number of layers dynamically from observation space
        self.num_layers = env.single_observation_space.shape[-1]

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Conv2d(self.num_layers, cnn_channels, 5, stride=3)
            ),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(
                nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1)
            ),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(cnn_channels, hidden_size // 2)),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.num_layers, hidden_size // 2)),
            nn.ReLU(),
        )

        # Get feature normalizations dynamically from environment
        feature_normalizations = getattr(env, "feature_normalizations", {})

        # Create normalization vector based on environment's feature normalizations
        # If no normalizations available, use ones as fallback
        if feature_normalizations:
            max_vec = torch.ones(self.num_layers, dtype=torch.float32)
            for feature_id, norm_value in feature_normalizations.items():
                if feature_id < self.num_layers:
                    max_vec[feature_id] = norm_value
        else:
            # Fallback to hardcoded values if feature_normalizations not available
            max_vec = torch.tensor(
                [
                    9.0,
                    1.0,
                    1.0,
                    10.0,
                    3.0,
                    254.0,
                    1.0,
                    1.0,
                    235.0,
                    8.0,
                    9.0,
                    250.0,
                    29.0,
                    1.0,
                    1.0,
                    8.0,
                    1.0,
                    1.0,
                    6.0,
                    3.0,
                    1.0,
                    2.0,
                ]
            )
            # Pad or truncate to match num_layers
            if len(max_vec) < self.num_layers:
                max_vec = torch.cat(
                    [max_vec, torch.ones(self.num_layers - len(max_vec))]
                )
            elif len(max_vec) > self.num_layers:
                max_vec = max_vec[: self.num_layers]

        max_vec = max_vec[None, :, None, None]  # Shape for broadcasting
        self.register_buffer("max_vec", max_vec)

        action_nvec = env.single_action_space.nvec
        self.actor = nn.ModuleList(
            [
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
                for n in action_nvec
            ]
        )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return (actions, value), hidden

    def encode_observations(self, observations, state=None):
        token_observations = observations
        B = token_observations.shape[0]
        TT = 1
        if token_observations.dim() != 3:  # hardcoding for shape [B, M, 3]
            TT = token_observations.shape[1]
            token_observations = einops.rearrange(
                token_observations, "b t m c -> (b t) m c"
            )

        assert token_observations.shape[-1] == 3, (
            f"Expected 3 channels per token. Got shape {token_observations.shape}"
        )
        token_observations[token_observations == 255] = 0

        # coords_byte contains x and y coordinates in a single byte (first 4 bits are x, last 4 bits are y)
        coords_byte = token_observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = token_observations[
            ..., 1
        ].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = token_observations[..., 2].float()  # Shape: [B_TT, M]

        # In ObservationShaper we permute. Here, we create the observations pre-permuted.
        # We'd like to pre-create this as part of initialization, but we don't know the batch size or time steps at
        # that point.
        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        batch_indices = (
            torch.arange(B * TT, device=token_observations.device)
            .unsqueeze(-1)
            .expand_as(atr_values)
        )

        # Add bounds checking to prevent out-of-bounds access
        valid_tokens = coords_byte != 0xFF
        valid_tokens = (
            valid_tokens
            & (x_coord_indices < self.out_width)
            & (y_coord_indices < self.out_height)
        )
        valid_tokens = valid_tokens & (
            atr_indices < self.num_layers
        )  # Also check attribute indices

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

        observations = box_obs

        # max_vec = box_obs.max(0)[0].max(1)[0].max(1)[0]
        # self.max_vec = torch.maximum(self.max_vec, max_vec[None, :, None, None])
        # if (np.random.rand() < 0.001):
        #    breakpoint()

        features = observations / self.max_vec
        # mmax = features.max(0)[0].max(1)[0].max(1)[0]
        # self.max_vec = torch.maximum(self.max_vec, mmax[None, :, None, None])
        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        return torch.cat([self_features, cnn_features], dim=1)

    def decode_actions(self, hidden):
        # hidden = self.layer_norm(hidden)
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value
