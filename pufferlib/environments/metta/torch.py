import einops
import numpy as np
import torch
from torch import nn

import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(
        self,
        env,
        policy=None,
        input_size=512,
        hidden_size=512,
        cnn_channels=128,
        **kwargs,
    ):
        if policy is None:
            policy = Policy(
                env,
                cnn_channels=cnn_channels,
                hidden_size=hidden_size,
                input_size=input_size,
            )
        super().__init__(env, policy, input_size, hidden_size)


class Policy(nn.Module):
    def __init__(
        self, env, cnn_channels=128, hidden_size=512, input_size=512, **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.is_continuous = False

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        # Use dynamic layer calculation to match metta agent implementations
        # This ensures we process all feature channels provided by the environment
        self.num_layers = max(env.feature_normalizations.keys()) + 1

        # Define CNN layers separately to calculate output size
        # Use gain=1.0 to match Metta's initialization (not sqrt(2))
        self.conv1 = pufferlib.pytorch.layer_init(
            nn.Conv2d(self.num_layers, cnn_channels, 5, stride=3), std=1.0
        )
        self.conv2 = pufferlib.pytorch.layer_init(
            nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1), std=1.0
        )

        # Calculate actual CNN output size dynamically
        test_input = torch.zeros(1, self.num_layers, self.out_width, self.out_height)
        with torch.no_grad():
            test_output = self.conv2(torch.relu(self.conv1(test_input)))
            self.cnn_flattened_size = test_output.numel() // test_output.shape[0]

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(
                nn.Linear(self.cnn_flattened_size, hidden_size // 2), std=1.0
            ),
            nn.ReLU(),
        )

        self.self_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(
                nn.Linear(self.num_layers, hidden_size // 2), std=1.0
            ),
            nn.ReLU(),
        )

        # Build normalization vector dynamically from environment feature_normalizations
        # This matches what Metta's fast policy and ObservationNormalizer do
        max_values = [1.0] * self.num_layers  # Default to 1.0
        for feature_id, norm_value in env.feature_normalizations.items():
            if feature_id < self.num_layers:
                max_values[feature_id] = norm_value if norm_value > 0 else 1.0

        max_vec = torch.tensor(max_values, dtype=torch.float32)
        # Clamp minimum value to 1.0 to avoid near-zero divisions
        max_vec = torch.maximum(max_vec, torch.ones_like(max_vec))
        max_vec = max_vec[None, :, None, None]
        self.register_buffer("max_vec", max_vec)

        action_nvec = env.single_action_space.nvec
        self.actor = nn.ModuleList(
            [
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
                for n in action_nvec
            ]
        )

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def encode_observations(
        self, observations: torch.Tensor, state=None
    ) -> torch.Tensor:
        """Converts raw observation tokens into a concatenated self + CNN feature vector."""
        B = observations.shape[0]
        TT = 1 if observations.dim() == 3 else observations.shape[1]

        if observations.dim() != 3:
            observations = einops.rearrange(observations, "b t m c -> (b t) m c")

        observations[observations == 255] = 0
        coords_byte = observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coords = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coords = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = observations[
            ..., 1
        ].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = observations[..., 2].float()  # Shape: [B_TT, M]

        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=observations.device,
        )

        valid_tokens = (
            (coords_byte != 0xFF)
            & (x_coords < self.out_width)
            & (y_coords < self.out_height)
            & (atr_indices < self.num_layers)
        )


        batch_idx = (
            torch.arange(B * TT, device=observations.device)
            .unsqueeze(-1)
            .expand_as(atr_values)
        )
        box_obs[
            batch_idx[valid_tokens],
            atr_indices[valid_tokens],
            x_coords[valid_tokens],
            y_coords[valid_tokens],
        ] = atr_values[valid_tokens]

        # Normalize features with epsilon for numerical stability
        features = box_obs / (self.max_vec + 1e-8)

        self_features = self.self_encoder(features[:, :, 5, 5])
        cnn_features = self.network(features)
        result = torch.cat([self_features, cnn_features], dim=1)
        return result

    def decode_actions(self, hidden):
        #hidden = self.layer_norm(hidden)
        logits = [dec(hidden) for dec in self.actor]
        value = self.value(hidden)
        return logits, value
