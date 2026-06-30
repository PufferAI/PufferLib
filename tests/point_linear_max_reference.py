import torch
import torch.nn as nn


def input_dim(self_dim, point_dim, num_points):
    return self_dim + num_points * point_dim


class PointLinearMaxEncoder(nn.Module):
    def __init__(self, self_dim=2, point_dim=4, num_points=16, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_obs_size = self_dim
        self.point_obs_size = point_dim
        self.num_points = num_points
        self.linear = nn.Linear(self.self_obs_size + self.point_obs_size, hidden_size)

    def forward(self, observations):
        observations = observations.float()
        point_obs = observations[:, self.self_obs_size:].reshape(
            observations.shape[0], self.num_points, self.point_obs_size)
        self_obs = observations[:, :self.self_obs_size].unsqueeze(1).expand(
            observations.shape[0], self.num_points, self.self_obs_size)
        point_inputs = torch.cat([self_obs, point_obs], dim=-1)
        return self.linear(point_inputs).max(dim=1)[0]


class FlatLinearEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, observations):
        return self.linear(observations.float())
