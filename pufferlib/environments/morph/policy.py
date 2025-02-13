import torch
from torch import nn
from pufferlib.pytorch import layer_init

import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Policy(nn.Module):
    def __init__(self, env, demo_size=358, hidden_size=512):
        super().__init__()
        self.is_continuous = True

        input_size = env.single_observation_space.shape[0]
        action_size = env.single_action_space.shape[0]

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        # NOTE: Original PHC network
        # self.actor_mlp = nn.Sequential(
        #     layer_init(nn.Linear(input_dim, 2048)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(2048, 1536)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(1536, 1024)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(1024, 1024)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(1024, 512)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(512, hidden)),
        #     nn.SiLU(),
        # )

        self.mu = nn.Sequential(
            # nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_size), std=0.01),
        )

        # NOTE: Original PHC uses a constant std. Something to experiment?
        self.sigma = nn.Parameter(
            torch.zeros(action_size, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )
        nn.init.constant_(self.sigma, -2.9)

        ### Separate Critic
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1)),
        )

        # NOTE: Original PHC network
        # self.critic_mlp = nn.Sequential(
        #     layer_init(nn.Linear(input_dim, 2048)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(2048, 1536)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(1536, 1024)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(1024, 1024)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(1024, 512)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(512, hidden)),
        #     nn.SiLU(),
        #     layer_init(nn.Linear(hidden, 1)),
        # )
        # self.value = nn.Linear(hidden, 1)

        ### Discriminator
        # NOTE: Check the demo_size from the env
        self._disc_mlp = nn.Sequential(
            layer_init(nn.Linear(demo_size, hidden_size)),
            nn.ReLU(),
            # layer_init(nn.Linear(1024, hidden)),
            # nn.ReLU(),
        )
        self._disc_logits = layer_init(torch.nn.Linear(hidden_size, 1))

        # NOTE: A hack to normalize the obs
        self.obs_mean = None

        self.obs_pointer = None

    def forward(self, observations):
        if self.obs_mean is None:
            self.obs_mean = torch.mean(observations, dim=0)
            self.obs_std = torch.std(observations, dim=0)

        observations = torch.clamp((observations - self.obs_mean) / self.obs_std, -10.0, 10.0)

        hidden, lookup = self.encode_observations(observations)
        actions, _ = self.decode_actions(hidden, lookup)
        value = self.critic_mlp(observations)
        return actions, value

    def encode_observations(self, obs):
        # Remember the obs to use in the critic
        self.obs_pointer = obs
        return self.actor_mlp(obs), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)
        probs = torch.distributions.Normal(mu, std)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value

    def discriminate(self, amp_obs):
        disc_mlp_out = self._disc_mlp(amp_obs)
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits

    def disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)

    def disc_weights(self):
        weights = []
        for m in self._disc_mlp.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self._disc_logits.weight))
        return weights
