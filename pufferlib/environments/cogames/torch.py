"""Torch policies for CoGames environments."""

import torch
import pufferlib.models
import pufferlib.pytorch


class Policy(pufferlib.models.Default):
    def __init__(self, env, hidden_size: int = 256, **kwargs):
        super().__init__(env, hidden_size=hidden_size)
        self.register_buffer("_inv_scale", torch.tensor(1.0 / 255.0), persistent=False)

    def encode_observations(self, observations, state=None):
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            obs_map = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            flattened = torch.cat([v.view(batch_size, -1) for v in obs_map.values()], dim=1)
        else:
            flattened = observations.view(batch_size, -1).float() * self._inv_scale
        return self.encoder(flattened)


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size: int = 256, hidden_size: int = 256):
        super().__init__(env, policy, input_size=input_size, hidden_size=hidden_size)
