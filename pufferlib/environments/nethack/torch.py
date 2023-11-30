from pdb import set_trace as T

import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib.models
import pufferlib.pytorch


class Recurrent:
    input_size = 512
    hidden_size = 512
    num_layers = 1

class Policy(pufferlib.models.Policy):
    '''Default NetHack Learning Environment policy ported from the nle release'''
    def __init__(self, env, *args,
            embedding_dim=32, crop_dim=9, num_layers=5,
            **kwargs):
        super().__init__(env)

        self.flat_observation_space = env.flat_observation_space
        self.flat_observation_structure = env.flat_observation_structure

        self.observation_shape = env.structured_observation_space
        self.glyph_shape = self.observation_shape["glyphs"].shape
        self.blstats_size = self.observation_shape["blstats"].shape[0]

        self.num_actions = env.action_space.n

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = 512

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        from nle import nethack
        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim**2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

        self.policy = nn.Linear(self.h_dim, self.num_actions)
        self.baseline = nn.Linear(self.h_dim, 1)

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def encode_observations(self, env_outputs):
        TB, _ = env_outputs.shape
        env_outputs = pufferlib.emulation.unpack_batched_obs(env_outputs,
            self.flat_observation_space, self.flat_observation_structure)

        glyphs = env_outputs["glyphs"].long()
        blstats = env_outputs["blstats"].float()

        # BL Stats
        coordinates = blstats[:, :2]
        blstats_emb = self.embed_blstats(blstats)
        assert blstats_emb.shape[0] == TB

        # Crop
        crop = self.crop(glyphs, coordinates)
        crop_emb = self._select(self.embed, crop).transpose(1, 3)
        crop_rep = self.extract_crop_representation(crop_emb).view(TB, -1)
        assert crop_rep.shape[0] == TB

        # Glyphs
        glyphs_emb = self._select(self.embed, glyphs).transpose(1, 3)
        glyphs_rep = self.extract_representation(glyphs_emb).view(TB, -1)
        assert glyphs_rep.shape[0] == TB

        st = torch.cat([blstats_emb, crop_rep, glyphs_rep], dim=1)
        st = self.fc(st)

        return st, None

    def decode_actions(self, hidden, lookup, concat=None):
        action = self.policy(hidden)
        baseline = self.baseline(hidden)
        return action, baseline

class Crop(nn.Module):
    """Helper class for NetHackNet below."""
    def __init__(self, height, width, height_target, width_target):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs, coordinates):
        """Calculates centered crop around given x,y coordinates.
        Args:
        inputs [B x H x W]
        coordinates [B x 2] x,y coordinates
        Returns:
        [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )

def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)
