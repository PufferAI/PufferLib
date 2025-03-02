import torch
from torch import nn
from pufferlib.pytorch import layer_init

import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Policy(nn.Module):
    def __init__(self, env, hidden_size=512, larger_critic=False):
        super().__init__()
        self.is_continuous = True

        input_size = env.single_observation_space.shape[0]
        action_size = env.single_action_space.shape[0]
        amp_obs_size = env.amp_observation_space.shape[0]

        self.obs_norm = torch.jit.script(RunningNorm(input_size))
        self.amp_obs_norm = torch.jit.script(RunningNorm(amp_obs_size))

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.SiLU(),
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
            nn.SiLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, action_size), std=0.01),
        )

        # NOTE: Original PHC uses a constant std. Something to experiment?
        self.sigma = nn.Parameter(
            torch.zeros(action_size, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )
        nn.init.constant_(self.sigma, -2.9)

        ### Separate Critic
        if larger_critic:
            self.critic_mlp = nn.Sequential(
                layer_init(nn.Linear(input_size, 2048)),
                nn.LayerNorm(2048),
                nn.ReLU(),
                layer_init(nn.Linear(2048, 1024)),
                nn.ReLU(),
                layer_init(nn.Linear(1024, 1024)),
                nn.ReLU(),
                layer_init(nn.Linear(1024, 512)),
                nn.ReLU(),
                layer_init(nn.Linear(512, 1), std=0.01),
            )

        else:
            self.critic_mlp = nn.Sequential(
                layer_init(nn.Linear(input_size, 1024)),
                nn.LayerNorm(1024),
                nn.ReLU(),
                layer_init(nn.Linear(1024, 1024)),
                nn.LayerNorm(1024),
                nn.ReLU(),
                layer_init(nn.Linear(1024, 512)),
                nn.LayerNorm(512),
                nn.ReLU(),
                layer_init(nn.Linear(512, 256)),
                nn.LayerNorm(256),
                nn.ReLU(),
                layer_init(nn.Linear(256, 1), std=0.01),
            )

        """
        # NOTE: Original PHC network
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(input_size, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.ReLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )
        """

        ### Discriminator
        # NOTE: Check the demo_size from the env
        self._disc_mlp = nn.Sequential(
            layer_init(nn.Linear(amp_obs_size, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.ReLU(),
        )
        self._disc_logits = layer_init(torch.nn.Linear(hidden_size, 1))

        self.obs_pointer = None
        self.mean_bound_loss = None

    def forward(self, observations):
        # if self.obs_mean is None:
        #     self.obs_mean = torch.mean(observations, dim=0)
        #     self.obs_std = torch.std(observations, dim=0)
        # observations = torch.clamp((observations - self.obs_mean) / self.obs_std, -10.0, 10.0)

        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, obs):
        # Remember the obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)
        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            self.mean_bound_loss = mean_violation.mean()

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value

    def discriminate(self, amp_obs):
        norm_amp_obs = self.amp_obs_norm(amp_obs)
        disc_mlp_out = self._disc_mlp(norm_amp_obs)
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits

    # def disc_logit_weights(self):
    #     return torch.flatten(self._disc_logits.weight)

    # def disc_weights(self):
    #     weights = []
    #     for m in self._disc_mlp.modules():
    #         if isinstance(m, nn.Linear):
    #             weights.append(torch.flatten(m.weight))

    #     weights.append(torch.flatten(self._disc_logits.weight))
    #     return weights

    def update_obs_rms(self, obs):
        self.obs_norm.update(obs)

    def update_amp_obs_rms(self, amp_obs):
        self.amp_obs_norm.update(amp_obs)

# This replaces gymnasium's NormalizeObservation wrapper
# NOTE: Tried BatchNorm1d with momentum=None, but the policy did not learn. Check again later.
# CHECK ME: To normalize obs, dividing by a constant is good, but each mujoco/brax env has a different scale...
class RunningNorm(nn.Module):
    def __init__(self, shape: int, epsilon=1e-5, clip=10.0):
        super().__init__()

        self.register_buffer("running_mean", torch.zeros((1, shape), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((1, shape), dtype=torch.float32))
        self.register_buffer("count", torch.ones(1, dtype=torch.float32))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        return torch.clamp(
            (x - self.running_mean.expand_as(x))
            / torch.sqrt(self.running_var.expand_as(x) + self.epsilon),
            -self.clip,
            self.clip,
        )

    @torch.jit.ignore
    def update(self, x):
        # NOTE: Separated update from forward to compile the policy
        # update() must be called to update the running mean and var
        with torch.no_grad():
            x = x.float()
            assert x.dim() == 2, "x must be 2D"
            mean = x.mean(0, keepdim=True)
            var = x.var(0, unbiased=False, keepdim=True)
            weight = 1 / self.count
            self.running_mean = self.running_mean * (1 - weight) + mean * weight
            self.running_var = self.running_var * (1 - weight) + var * weight
            self.count += 1

    # NOTE: below are needed to torch.save() the model
    @torch.jit.ignore
    def __getstate__(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    @torch.jit.ignore
    def __setstate__(self, state):
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]
        self.clip = state["clip"]
