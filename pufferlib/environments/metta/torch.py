import numpy as np
import torch
from torch import nn
from tensordict import TensorDict

import pufferlib.models
from metta.agent.pytorch.fast import Fast


class Policy(nn.Module):
    """Policy wrapper around Metta's Fast policy for Pufferlib integration."""

    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core Fast policy
        self.fast_policy = Fast(
            env, input_size=input_size, hidden_size=hidden_size, **kwargs
        )

        self._initialize_to_environment(env)

    def _initialize_to_environment(self, env):
        """Initialize Fast policy tensors and mappings from the environment."""

        # Cumulative action max params (like MettaAgent does)
        cum_action_max_params = torch.tensor(
            [0] + list(np.cumsum(env.max_action_args)),
            dtype=torch.int32,
            device=self.device,
        )

        # Action index tensor
        action_index_tensor = torch.tensor(
            [
                [idx, j]
                for idx, max_param in enumerate(env.max_action_args)
                for j in range(max_param + 1)
            ],
            dtype=torch.int32,
            device=self.device,
        )

        full_action_names = [
            f"{name}_{i}"
            for name, max_param in zip(env.action_names, env.max_action_args, strict=False)
            for i in range(max_param + 1)
        ]

        # Initialize Fast policy
        self.fast_policy.initialize_to_environment(full_action_names, self.device)

        # Share tensors with Fast policy (MettaAgent style)
        self.fast_policy.action_index_tensor = action_index_tensor
        self.fast_policy.cum_action_max_params = cum_action_max_params

    def forward_training(self, observations, action, state=None):
        """Forward pass during training (with action)."""
        self._maybe_reset_memory(observations)

        td = TensorDict({"env_obs": observations}, batch_size=observations.shape[0])
        state = self._sanitize_state(state)

        result_td = self.fast_policy(td, state=state, action=action)

        logits = result_td["full_log_probs"]
        value = result_td["values"]
        entropy = result_td.get("entropy")

        return [logits], value, entropy

    def forward_eval(self, observations, state=None):
        """Forward pass during evaluation (no action)."""
        self._maybe_reset_memory(observations)

        td = TensorDict({"env_obs": observations}, batch_size=observations.shape[0])
        result_td = self.fast_policy(td, state=state, action=None)

        logits = result_td["full_log_probs"]  # Shape: [batch_size, total_actions]
        values = result_td["values"]

        # Pufferlib expects a list for multi-discrete compatibility
        return [logits], values

    def forward(self, observations, state=None):
        """Default forward (inference)."""
        logits, value, _ = self.forward_training(observations, action=None, state=state)
        return logits, value

    def _maybe_reset_memory(self, observations):
        """Reset LSTM memory if batch size mismatch occurs."""
        lstm_h = self.fast_policy.lstm_h
        if lstm_h and list(lstm_h.values())[0].shape[0] != observations.shape[0]:
            self.fast_policy.reset_memory()

    def _sanitize_state(self, state):
        """Remove incompatible entries from state (e.g., action TensorDict)."""
        if state is None or not isinstance(state, dict):
            return state

        return {
            k: v
            for k, v in state.items()
            if k in {"lstm_h", "lstm_c", "hidden"}  # keep LSTM state
        }

    def to(self, device):
        """Ensure action tensors move with the module."""
        result = super().to(device)

        for tensor_name in ("action_index_tensor", "cum_action_max_params"):
            tensor = getattr(self.fast_policy, tensor_name, None)
            if tensor is not None:
                setattr(self.fast_policy, tensor_name, tensor.to(device))

        return result
