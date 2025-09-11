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

        # Store action space info for logits splitting
        self.action_nvec = env.single_action_space.nvec
        
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

    def _split_logits_for_multidiscrete(self, flattened_logits):
        """Split flattened logits into separate tensors for each action dimension.
        
        The Fast policy outputs flattened logits with shape [batch_size, total_actions].
        We use the action_index_tensor to correctly map flattened indices to MultiDiscrete actions.
        """
        # Use the same mapping that Fast policy uses
        action_index_tensor = self.fast_policy.action_index_tensor  # Shape: [total_actions, 2]
        
        batch_size = flattened_logits.shape[0]
        num_action_types = self.action_nvec[0]  # First dimension: action types  
        max_action_params = self.action_nvec[1]  # Second dimension: action params
        
        # Collect logits for each action type and param (avoid in-place operations)
        action_type_logits_list = [[] for _ in range(num_action_types)]
        action_param_logits_list = [[] for _ in range(max_action_params)]
        
        # Use the action_index_tensor to map flattened logits to MultiDiscrete logits
        for flat_idx in range(min(action_index_tensor.shape[0], flattened_logits.shape[1])):
            action_type, action_param = action_index_tensor[flat_idx].tolist()
            current_logit = flattened_logits[:, flat_idx]
            
            # Collect logits for this action type
            if action_type < num_action_types:
                action_type_logits_list[action_type].append(current_logit)
            
            # Collect logits for this action param
            if action_param < max_action_params:
                action_param_logits_list[action_param].append(current_logit)
        
        # Aggregate collected logits using logsumexp (non-in-place)
        action_type_logits = torch.zeros(batch_size, num_action_types, 
                                        device=flattened_logits.device, dtype=flattened_logits.dtype)
        action_param_logits = torch.zeros(batch_size, max_action_params,
                                         device=flattened_logits.device, dtype=flattened_logits.dtype)
        
        for action_type in range(num_action_types):
            if action_type_logits_list[action_type]:
                stacked_logits = torch.stack(action_type_logits_list[action_type], dim=1)
                action_type_logits[:, action_type] = torch.logsumexp(stacked_logits, dim=1)
            else:
                action_type_logits[:, action_type] = float('-inf')
                
        for action_param in range(max_action_params):
            if action_param_logits_list[action_param]:
                stacked_logits = torch.stack(action_param_logits_list[action_param], dim=1)
                action_param_logits[:, action_param] = torch.logsumexp(stacked_logits, dim=1)
            else:
                action_param_logits[:, action_param] = float('-inf')
                
        return [action_type_logits, action_param_logits]

    def forward_training(self, observations, action, state=None):
        """Forward pass during training (with action)."""
        self._maybe_reset_memory(observations)

        td = TensorDict({"env_obs": observations}, batch_size=observations.shape[0])
        state = self._sanitize_state(state)

        result_td = self.fast_policy(td, state=state, action=action)

        flattened_logits = result_td["full_log_probs"]
        value = result_td["values"]
        entropy = result_td.get("entropy")

        # Split flattened logits into separate action dimensions
        logits_list = self._split_logits_for_multidiscrete(flattened_logits)

        return logits_list, value, entropy

    def forward_eval(self, observations, state=None):
        """Forward pass during evaluation (no action)."""
        self._maybe_reset_memory(observations)

        td = TensorDict({"env_obs": observations}, batch_size=observations.shape[0])
        result_td = self.fast_policy(td, state=state, action=None)

        flattened_logits = result_td["full_log_probs"]  # Shape: [batch_size, total_actions]
        values = result_td["values"]

        # Split flattened logits into separate action dimensions for MultiDiscrete
        logits_list = self._split_logits_for_multidiscrete(flattened_logits)
        
        return logits_list, values

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
