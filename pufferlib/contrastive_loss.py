import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict


class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning in PufferLib.

    Implements InfoNCE loss with geometric future positives and shuffled negatives.
    The loss samples (st, at) pairs, creates positive examples sf^(1) by looking
    Δ ~ GEOM(1-γ) steps ahead, and generates negative examples by shuffling.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        contrastive_coef: float = 1.0,
        embedding_dim: int = 256,
        discount: float = 0.99,
        use_projection_head: bool = False,
        device: torch.device = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.contrastive_coef = contrastive_coef
        self.embedding_dim = embedding_dim
        self.discount = discount
        self.device = device or torch.device('cpu')

        # Projection head will be created dynamically if needed
        self.projection_head = None
        self.use_projection_head = use_projection_head
        self._value_projection = None

        # Metrics tracking
        self.loss_tracker = defaultdict(list)

    def forward(
        self,
        embeddings: torch.Tensor,
        terminals: torch.Tensor,
        truncations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute contrastive loss.

        Args:
            embeddings: [segments, horizon, embedding_dim] tensor of representations
            terminals: [segments, horizon] tensor of done flags
            truncations: [segments, horizon] tensor of truncation flags (optional)

        Returns:
            loss: Contrastive loss tensor
            metrics: Dictionary of metrics for logging
        """
        segments, horizon = embeddings.shape[0], embeddings.shape[1]
        embedding_dim = embeddings.shape[-1]

        if embedding_dim == 0:
            return torch.tensor(0.0, device=self.device), self._empty_metrics()

        # Create done mask combining terminals and truncations
        done_mask = terminals.to(dtype=torch.bool)
        if truncations is not None:
            done_mask = torch.logical_or(done_mask, truncations.to(dtype=torch.bool))

        # Apply projection head if configured
        if self.use_projection_head:
            embeddings = self._apply_projection_head(embeddings)

        # Sample contrastive pairs
        batch_indices, anchor_steps, positive_steps, sampled_deltas = self._sample_pairs(
            done_mask, segments, horizon
        )

        num_pairs = len(batch_indices)
        if num_pairs < 2:
            return torch.tensor(0.0, device=self.device), self._empty_metrics(num_pairs)

        # Extract embeddings for contrastive learning
        batch_idx_tensor = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
        anchor_idx_tensor = torch.tensor(anchor_steps, device=self.device, dtype=torch.long)
        positive_idx_tensor = torch.tensor(positive_steps, device=self.device, dtype=torch.long)

        anchor_embeddings = embeddings[batch_idx_tensor, anchor_idx_tensor]
        positive_embeddings = embeddings[batch_idx_tensor, positive_idx_tensor]

        # Compute similarities and InfoNCE loss
        similarities = anchor_embeddings @ positive_embeddings.T
        positive_logits = similarities.diagonal().unsqueeze(1)

        # Create negative logits by masking out positive pairs
        mask = torch.eye(num_pairs, device=self.device, dtype=torch.bool)
        negative_logits = similarities[~mask].view(num_pairs, num_pairs - 1)

        # Combine positive and negative logits
        logits = torch.cat([positive_logits, negative_logits], dim=1) / self.temperature
        labels = torch.zeros(num_pairs, dtype=torch.long, device=self.device)

        # Compute InfoNCE loss
        infonce_loss = F.cross_entropy(logits, labels, reduction="mean")

        # Compute metrics
        metrics = self._compute_metrics(
            positive_logits, negative_logits, num_pairs, sampled_deltas
        )

        return infonce_loss * self.contrastive_coef, metrics

    def _apply_projection_head(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply projection head to embeddings."""
        if self.projection_head is None:
            input_dim = embeddings.shape[-1]
            self.projection_head = nn.Linear(input_dim, self.embedding_dim).to(self.device)

        # Reshape for linear layer: [segments, horizon, dim] -> [segments*horizon, dim]
        original_shape = embeddings.shape[:2]
        embeddings_flat = embeddings.view(-1, embeddings.shape[-1])
        projected_flat = self.projection_head(embeddings_flat)

        # Reshape back: [segments*horizon, embedding_dim] -> [segments, horizon, embedding_dim]
        return projected_flat.view(*original_shape, self.embedding_dim)

    def _sample_pairs(
        self,
        done_mask: torch.Tensor,
        segments: int,
        horizon: int
    ) -> Tuple[list, list, list, list]:
        """Sample anchor and positive pairs using geometric distribution."""
        prob = max(1.0 - float(self.discount), 1e-8)
        geom_dist = torch.distributions.Geometric(
            probs=torch.tensor(prob, device=self.device)
        )

        done_mask_cpu = done_mask.detach().to("cpu")

        batch_indices = []
        anchor_steps = []
        positive_steps = []
        sampled_deltas = []

        for batch_idx in range(segments):
            done_row = done_mask_cpu[batch_idx].view(-1)

            # Find episode boundaries
            episode_bounds = []
            start = 0
            for step, done in enumerate(done_row.tolist()):
                if done:
                    episode_bounds.append((start, step))
                    start = step + 1
            if start < horizon:
                episode_bounds.append((start, horizon - 1))

            # Collect candidate anchors
            candidate_anchors = []
            for episode_start, episode_end in episode_bounds:
                if episode_end - episode_start < 1:
                    continue
                for anchor in range(episode_start, episode_end):
                    candidate_anchors.append((anchor, episode_end))

            if not candidate_anchors:
                continue

            # Sample anchor and positive
            choice_idx = int(torch.randint(len(candidate_anchors), (1,), device=self.device).item())
            anchor_step, episode_end = candidate_anchors[choice_idx]
            max_future = episode_end - anchor_step

            if max_future < 1:
                continue

            # Sample delta using geometric distribution
            delta = int(geom_dist.sample().item())
            attempts = 0
            while delta > max_future and attempts < 10:
                delta = int(geom_dist.sample().item())
                attempts += 1
            if delta > max_future:
                delta = max_future

            positive_step = anchor_step + delta

            batch_indices.append(batch_idx)
            anchor_steps.append(anchor_step)
            positive_steps.append(positive_step)
            sampled_deltas.append(float(delta))

        return batch_indices, anchor_steps, positive_steps, sampled_deltas

    def _compute_metrics(
        self,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        num_pairs: int,
        sampled_deltas: list,
    ) -> Dict[str, float]:
        """Compute metrics for logging."""
        return {
            "positive_sim_mean": positive_logits.mean().item(),
            "negative_sim_mean": negative_logits.mean().item(),
            "positive_sim_std": positive_logits.std().item(),
            "negative_sim_std": negative_logits.std().item(),
            "num_pairs": num_pairs,
            "delta_mean": float(sum(sampled_deltas) / len(sampled_deltas)) if sampled_deltas else 0.0,
        }

    def _empty_metrics(self, num_pairs: int = 0) -> Dict[str, float]:
        """Return empty metrics when no loss can be computed."""
        return {
            "positive_sim_mean": 0.0,
            "negative_sim_mean": 0.0,
            "positive_sim_std": 0.0,
            "negative_sim_std": 0.0,
            "num_pairs": num_pairs,
            "delta_mean": 0.0,
        }


def get_embeddings_from_policy_data(
    policy_logits: torch.Tensor,
    policy_values: torch.Tensor,
    embedding_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract embeddings from policy outputs.

    This is a helper function to extract embeddings when they're not directly
    available from the policy. In practice, you'd want to modify your policy
    to return embeddings directly.

    Args:
        policy_logits: Action logits from policy forward pass
        policy_values: Value predictions from policy
        embedding_dim: Desired embedding dimension
        device: Target device

    Returns:
        embeddings: [batch_size, embedding_dim] tensor
    """
    # Fallback: use value as embeddings but create learnable projection
    # This is suboptimal but demonstrates the interface
    values = policy_values.squeeze(-1) if policy_values.dim() > 1 else policy_values

    if values.dim() == 1:
        # Create a simple learnable projection from 1D value to embedding_dim
        projection = nn.Linear(1, embedding_dim).to(device)
        nn.init.xavier_uniform_(projection.weight)
        values = values.unsqueeze(-1)  # [N] -> [N, 1]
        embeddings = projection(values)  # [N, 1] -> [N, embedding_dim]
        return embeddings
    else:
        return values


def compute_contrastive_loss_pufferlib(
    embeddings: torch.Tensor,
    terminals: torch.Tensor,
    truncations: Optional[torch.Tensor] = None,
    temperature: float = 0.1,
    contrastive_coef: float = 1.0,
    embedding_dim: int = 256,
    discount: float = 0.99,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Functional interface for contrastive loss computation.

    This function can be directly integrated into PufferLib's training loop
    without needing to modify the main PuffeRL class.

    Args:
        embeddings: [segments, horizon, embedding_dim] representation tensor
        terminals: [segments, horizon] done flags tensor
        truncations: [segments, horizon] truncation flags (optional)
        temperature: Temperature for InfoNCE loss
        contrastive_coef: Coefficient for contrastive loss
        embedding_dim: Target embedding dimension
        discount: Discount factor for geometric sampling
        device: Target device

    Returns:
        loss: Contrastive loss value
        metrics: Dictionary of logging metrics
    """
    contrastive_loss = ContrastiveLoss(
        temperature=temperature,
        contrastive_coef=contrastive_coef,
        embedding_dim=embedding_dim,
        discount=discount,
        device=device or embeddings.device,
    )

    return contrastive_loss(embeddings, terminals, truncations)