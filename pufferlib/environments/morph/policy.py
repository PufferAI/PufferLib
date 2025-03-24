import torch
from torch import nn

from pufferlib.pytorch import layer_init
import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

        # Point to the original policy's methods
        self.set_deterministic_action = self.policy.set_deterministic_action
        self.discriminate = self.policy.discriminate
        self.update_obs_rms = self.policy.update_obs_rms
        self.update_amp_obs_rms = self.policy.update_amp_obs_rms

    @property
    def mean_bound_loss(self):
        return self.policy.mean_bound_loss


class PolicyWithDiscriminator(nn.Module):
    def __init__(self, env, hidden_size=512):
        super().__init__()
        self.is_continuous = True
        self._deterministic_action = False

        self.input_size = env.single_observation_space.shape[0]
        self.action_size = env.single_action_space.shape[0]

        # Assume the action space is symmetric (low=-high)
        self.soft_bound = 0.9 * env.single_action_space.high[0]

        self.obs_norm = torch.jit.script(RunningNorm(self.input_size))

        ### Actor
        self.actor_mlp = None
        self.mu = nn.Sequential(
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        # NOTE: Original PHC uses a constant std. Something to experiment?
        self.sigma = nn.Parameter(
            torch.zeros(self.action_size, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )
        nn.init.constant_(self.sigma, -2.9)

        ### Critic
        self.critic_mlp = None

        ### Discriminator
        self.use_amp_obs = env.amp_observation_space is not None
        self.amp_obs_norm = None

        if self.use_amp_obs:
            amp_obs_size = env.amp_observation_space.shape[0]
            self.amp_obs_norm = torch.jit.script(RunningNorm(amp_obs_size))

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
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, obs):
        raise NotImplementedError

    def decode_actions(self, hidden, lookup=None):
        raise NotImplementedError

    def set_deterministic_action(self, value):
        self._deterministic_action = value

    def discriminate(self, amp_obs):
        if not self.use_amp_obs:
            return None

        norm_amp_obs = self.amp_obs_norm(amp_obs)
        disc_mlp_out = self._disc_mlp(norm_amp_obs)
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits

    # NOTE: Used for network weight regularization
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
        if not self.use_amp_obs:
            return

        self.amp_obs_norm.update(amp_obs)

    def bound_loss(self, mu):
        mu_loss = torch.zeros_like(mu)
        mu_loss = torch.where(mu > self.soft_bound, (mu - self.soft_bound) ** 2, mu_loss)
        mu_loss = torch.where(mu < -self.soft_bound, (mu + self.soft_bound) ** 2, mu_loss)
        return mu_loss.mean()


# NOTE: The PHC implementation, which has no LSTM. 17.0M params
class PHCPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        # NOTE: Original PHC network + LayerNorm
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        # NOTE: Original PHC network + LayerNorm
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value


class LSTMCriticPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        # Actor: Original PHC network
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )
        self.mu = None

        ### Critic with LSTM
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.ReLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the actor
        self.obs_pointer = self.obs_norm(obs)

        # NOTE: hidden goes through LSTM, then to the value (critic head)
        return self.critic_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.actor_mlp(self.obs_pointer)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: hidden from LSTM goes to the critic head
        value = self.value(hidden)
        return probs, value


# NOTE: 13.5M params, Worked for simple motions, but not capable for many, complex motions
class LSTMActorPolicy(PolicyWithDiscriminator):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.SiLU(),
        )

        self.mu = nn.Sequential(
            nn.SiLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            # nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            # nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value


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
            (x - self.running_mean.expand_as(x)) / torch.sqrt(self.running_var.expand_as(x) + self.epsilon),
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
