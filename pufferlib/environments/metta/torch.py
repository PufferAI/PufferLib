import numpy as np
import torch
from torch import nn

import pufferlib.models
from metta.agent.pytorch.fast import Fast


class Policy(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()

        # Create the Metta Fast policy
        self.fast_policy = Fast(env, input_size=input_size, hidden_size=hidden_size, **kwargs)

        # Initialize the policy to the environment
        # Get the actual MettaGrid environment from the wrapper
        metta_env = self._get_metta_env(env)
        if metta_env is not None:
            action_names = metta_env.action_names
            max_action_args = metta_env.max_action_args
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Replicate MettaAgent.initialize_to_environment() logic

            # Create cumulative action max params (like MettaAgent does)
            cum_action_max_params = torch.tensor(
                [0] + list(np.cumsum(max_action_args)), dtype=torch.int32, device=device
            )

            # Create action index tensor
            action_index_tensor = torch.tensor(
                [[idx, j] for idx, max_param in enumerate(max_action_args) for j in range(max_param + 1)],
                device=device,
                dtype=torch.int32,
            )

            # Generate full action names
            full_action_names = [
                f"{name}_{i}"
                for name, max_param in zip(action_names, max_action_args, strict=False)
                for i in range(max_param + 1)
            ]

            # Initialize the Fast policy to the environment (mixin method signature)
            self.fast_policy.initialize_to_environment(full_action_names, device)

            # Share tensors with policy (like MettaAgent does)
            self.fast_policy.action_index_tensor = action_index_tensor
            self.fast_policy.cum_action_max_params = cum_action_max_params

    def _get_metta_env(self, env):
        """Extract the MettaGrid environment from PufferLib wrappers."""
        # Try to find the underlying MettaGrid environment
        current_env = env

        # Look for common attributes that indicate we have a MettaGrid environment
        while hasattr(current_env, "_env") or hasattr(current_env, "env"):
            if hasattr(current_env, "get_observation_features"):
                return current_env

            # Try _env first, then env
            if hasattr(current_env, "_env"):
                current_env = current_env._env
            elif hasattr(current_env, "env"):
                current_env = current_env.env
            else:
                break

        # Check if current_env has the required methods
        if hasattr(current_env, "get_observation_features"):
            return current_env

        return None

    def forward_eval(self, observations, state=None):
        # Convert observations to TensorDict format expected by Metta Fast policy
        from tensordict import TensorDict

        # Create TensorDict with proper structure
        td = TensorDict(
            {
                "env_obs": observations,
                "env_id": torch.arange(observations.shape[0], device=observations.device),
            },
            batch_size=observations.shape[0],
        )

        # Forward through Metta Fast policy
        result_td = self.fast_policy(td)

        # Return Metta's flat full_log_probs and values directly. The
        # environment `single_action_space` has been adjusted to expose a
        # flattened Discrete space so pufferlib's sampling expects a single
        # integer action that indexes into this flat vector.
        logits = result_td["full_log_probs"]  # Shape: [batch_size, total_actions]
        values = result_td["values"]

        # Return a single-element list so pufferlib's `sample_logits`
        # treats this as a (multi-discrete) list of argument logits and
        # produces an action of shape [batch_size, 1], matching the
        # environment's `single_action_space` of shape (1,).
        return [logits], values

    def forward(self, observations, state=None):
        # Training path: pufferl passes a `state` dict (may contain lstm_h/lstm_c).
        # To avoid mismatches with Metta's internal LSTM batching, handle the
        # non-recurrent training path by calling the inner policy's
        # encode/decode directly (no LSTM). For evaluation/inference we
        # continue to use the full Metta Fast forward.
        if state is not None:
            # Use the policy's encode/decode pipeline without running the LSTM
            # to produce logits and values compatible with pufferlib training.
            try:
                # The Fast instance stores the underlying Policy at `policy`.
                inner_policy = getattr(self.fast_policy, "policy", None)
                if inner_policy is None:
                    # Fallback to full forward
                    return self.forward_eval(observations, state)

                # Encode observations to hidden (matches Fast.encode_observations)
                hidden = inner_policy.encode_observations(observations, state)

                # hidden may be shaped (B, ...) or (B, TT, -1). Ensure flattened
                # shape expected by decode_actions: (batch_size, hidden_size)
                if hidden.dim() > 2:
                    flat_hidden = hidden.reshape(hidden.shape[0], -1)
                    batch_size = flat_hidden.shape[0]
                else:
                    flat_hidden = hidden
                    batch_size = flat_hidden.shape[0]

                logits, value = inner_policy.decode_actions(flat_hidden, batch_size)
                return logits, value
            except Exception:
                # On any error, fallback to evaluation forward which uses the
                # full Fast.forward that handles recurrent cases.
                return self.forward_eval(observations, state)

        return self.forward_eval(observations, state)

    def to(self, device):
        """Override to method to ensure action tensors are moved with the policy."""
        result = super().to(device)

        # Move action tensors to the same device if they exist
        if hasattr(self.fast_policy, "action_index_tensor") and self.fast_policy.action_index_tensor is not None:
            self.fast_policy.action_index_tensor = self.fast_policy.action_index_tensor.to(device)
        if hasattr(self.fast_policy, "cum_action_max_params") and self.fast_policy.cum_action_max_params is not None:
            self.fast_policy.cum_action_max_params = self.fast_policy.cum_action_max_params.to(device)

        return result


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy=None, input_size=128, hidden_size=128, **kwargs):
        if policy is None:
            policy = Policy(env, input_size=input_size, hidden_size=hidden_size, **kwargs)
        super().__init__(env, policy, input_size, hidden_size)
