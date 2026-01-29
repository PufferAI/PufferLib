"""Checkpoint Queue for Self-Play Training.

Manages a queue of policy checkpoints where the opponent is always N checkpoints
behind the learner. This creates a stable skill gap and natural curriculum.

Training Flow:
    Stages 0-9:   Autopilot opponent (curriculum)
    Stage 10:     Save checkpoint A (milestone)
    Stages 10-19: Continue curriculum with autopilot
    Stage 20:     Save checkpoint B (milestone), START SELF-PLAY vs A
    Dominate:     Save checkpoint C, upgrade opponent to B
    Dominate:     Save checkpoint D, upgrade opponent to C
    ...and so on (opponent always `lag` checkpoints behind)

Lag Semantics:
    lag=1 means "2nd newest" (skip 1 checkpoint):
    - Queue: [A, B, C] with lag=1 -> opponent uses B (index -2)
    - Queue: [A, B, C, D] with lag=1 -> opponent uses C (index -2)
"""
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class QueueEntry:
    """A checkpoint in the queue."""
    path: str           # Checkpoint file path
    step: int           # Global step when saved
    stage: float        # Curriculum stage when saved
    tag: str            # "stage10", "stage20", or "selfplay_N"

    def is_milestone(self) -> bool:
        """Return True if this is a milestone checkpoint (stage10/stage20)."""
        return self.tag in ("stage10", "stage20")


class CheckpointQueue:
    """Manages checkpoint queue for self-play training.

    Checkpoints are saved when the learner dominates the opponent (exceeds
    perf_threshold). The opponent is loaded from an older checkpoint in the
    queue (determined by lag parameter).

    Milestone checkpoints (stage10, stage20) are never pruned.
    """

    def __init__(self, save_dir: str, max_checkpoints: int = 20):
        """Initialize checkpoint queue.

        Args:
            save_dir: Directory to store checkpoint files
            max_checkpoints: Maximum selfplay checkpoints to keep (milestones always kept)
        """
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[QueueEntry] = []

        # Create save directory if needed
        os.makedirs(save_dir, exist_ok=True)

        print(f'[CHECKPOINT-QUEUE] Initialized: save_dir={save_dir}, max={max_checkpoints}')

    def save(self, policy, step: int, stage: float, tag: str) -> str:
        """Save checkpoint and add to queue.

        Args:
            policy: PyTorch policy module to save
            step: Current global step
            stage: Current curriculum stage
            tag: Checkpoint tag ("stage10", "stage20", or "selfplay_N")

        Returns:
            Path to saved checkpoint file
        """
        # Generate filename
        filename = f"checkpoint_{tag}_step{step}.pt"
        path = os.path.join(self.save_dir, filename)

        # Save checkpoint
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'step': step,
            'stage': stage,
            'tag': tag,
        }, path)

        # Add to queue
        entry = QueueEntry(path=path, step=step, stage=stage, tag=tag)
        self.checkpoints.append(entry)

        print(f'[CHECKPOINT-QUEUE] Saved {tag} at step {step}: {path}')

        # Prune old checkpoints if needed
        self._prune_old_checkpoints()

        return path

    def get_opponent(self, lag: int = 1) -> Optional[str]:
        """Get checkpoint path for opponent.

        Args:
            lag: How many positions behind the latest (1=2nd newest, index -2)

        Returns:
            Path to opponent checkpoint, or None if queue too small
        """
        if len(self.checkpoints) < lag + 1:
            return None

        # lag=1 means index -2 (2nd newest)
        index = -(lag + 1)
        return self.checkpoints[index].path

    def get_opponent_entry(self, lag: int = 1) -> Optional[QueueEntry]:
        """Get full QueueEntry for opponent.

        Args:
            lag: How many positions behind the latest (1=2nd newest, index -2)

        Returns:
            QueueEntry for opponent, or None if queue too small
        """
        if len(self.checkpoints) < lag + 1:
            return None

        index = -(lag + 1)
        return self.checkpoints[index]

    def should_upgrade(self, current_opponent_path: Optional[str], lag: int) -> Optional[str]:
        """Check if opponent should be upgraded to newer checkpoint.

        Args:
            current_opponent_path: Path to current opponent checkpoint
            lag: Desired lag positions behind latest

        Returns:
            New opponent path if upgrade needed, None otherwise
        """
        new_path = self.get_opponent(lag)

        if new_path is None:
            return None

        if new_path != current_opponent_path:
            return new_path

        return None

    def _prune_old_checkpoints(self):
        """Remove oldest selfplay checkpoints, keeping milestones forever."""
        # Count selfplay checkpoints (not milestones)
        selfplay_checkpoints = [c for c in self.checkpoints if not c.is_milestone()]

        # Reserve 2 slots for milestones (stage10, stage20)
        max_selfplay = self.max_checkpoints - 2

        while len(selfplay_checkpoints) > max_selfplay:
            # Find oldest selfplay checkpoint
            oldest = selfplay_checkpoints.pop(0)

            # Remove file
            if os.path.exists(oldest.path):
                try:
                    os.remove(oldest.path)
                    print(f'[CHECKPOINT-QUEUE] Pruned old checkpoint: {oldest.path}')
                except OSError as e:
                    print(f'[CHECKPOINT-QUEUE] Warning: Could not remove {oldest.path}: {e}')

            # Remove from main list
            self.checkpoints.remove(oldest)

    def __len__(self) -> int:
        """Return number of checkpoints in queue."""
        return len(self.checkpoints)

    def __repr__(self) -> str:
        """Return string representation of queue."""
        tags = [c.tag for c in self.checkpoints]
        return f"CheckpointQueue({tags})"

    def get_queue_state(self) -> dict:
        """Get serializable state of the queue for logging/debugging."""
        return {
            'num_checkpoints': len(self.checkpoints),
            'tags': [c.tag for c in self.checkpoints],
            'steps': [c.step for c in self.checkpoints],
            'milestones': [c.tag for c in self.checkpoints if c.is_milestone()],
        }
