import numpy as np
import torch
from torch import nn

import pufferlib.models
from metta.agent.pytorch.fast import Fast


class Policy(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()

        # Create the Metta Fast policy
        self.fast_policy = Fast(
            env, input_size=input_size, hidden_size=hidden_size, **kwargs
        )

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
                [
                    [idx, j]
                    for idx, max_param in enumerate(max_action_args)
                    for j in range(max_param + 1)
                ],
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

    def forward_training(self, observations, action, state=None):
        # Convert observations to TensorDict format expected by Metta Fast policy
        from tensordict import TensorDict

        lstm_h = self.fast_policy.lstm_h
        if lstm_h:
            if list(lstm_h.values())[0].shape[0] != observations.shape[0]:
                self.fast_policy.reset_memory()

        # Create TensorDict with proper structure
        td = TensorDict(
            {
                "env_obs": observations,
            },
            batch_size=observations.shape[0],
        )

        # Handle state TensorDict batch size mismatch
        if state is not None and isinstance(state, dict):
            # Create a clean state dict with only LSTM state, excluding action TensorDict
            clean_state = {}
            for key, value in state.items():
                if key in ["lstm_h", "lstm_c", "hidden"]:
                    # Keep LSTM state as-is, let fast policy handle batch size
                    clean_state[key] = value
                elif key == "action":
                    # Skip the action TensorDict that's causing batch size issues
                    continue
                else:
                    clean_state[key] = value
            state = clean_state

        # Forward through Metta Fast policy
        result_td = self.fast_policy(td, state=state, action=action)

        logits = result_td["full_log_probs"]
        value = result_td["values"]
        entropy = result_td.get("entropy", None)

        return [logits], value, entropy

    def forward_eval(self, observations, state=None):
        # Convert observations to TensorDict format expected by Metta Fast policy
        from tensordict import TensorDict

        lstm_h = self.fast_policy.lstm_h
        if lstm_h:
            if list(lstm_h.values())[0].shape[0] != observations.shape[0]:
                self.fast_policy.reset_memory()

        # Create TensorDict with proper structure
        td = TensorDict(
            {
                "env_obs": observations,
            },
            batch_size=observations.shape[0],
        )

        # Forward through Metta Fast policy
        result_td = self.fast_policy(td, state=state, action=None)

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
        # For inference, let the fast policy handle action sampling
        # Don't pass action from state to avoid batch size mismatches
        logits, value, entropy = self.forward_training(
            observations, action=None, state=state
        )
        return logits, value

    def to(self, device):
        """Override to method to ensure action tensors are moved with the policy."""
        result = super().to(device)

        # Move action tensors to the same device if they exist
        if (
            hasattr(self.fast_policy, "action_index_tensor")
            and self.fast_policy.action_index_tensor is not None
        ):
            self.fast_policy.action_index_tensor = (
                self.fast_policy.action_index_tensor.to(device)
            )
        if (
            hasattr(self.fast_policy, "cum_action_max_params")
            and self.fast_policy.cum_action_max_params is not None
        ):
            self.fast_policy.cum_action_max_params = (
                self.fast_policy.cum_action_max_params.to(device)
            )

        return result
