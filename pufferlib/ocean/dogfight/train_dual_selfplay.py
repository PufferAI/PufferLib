#!/usr/bin/env python
"""Dual-Perspective Self-Play Training for Dogfight with Checkpoint Queue.

This script trains a single policy on BOTH perspectives of each dogfight episode:
- Player perspective: (obs, actions, rewards)
- Opponent perspective: (obs, actions, -rewards)

The opponent is loaded from a checkpoint queue, always N checkpoints behind the learner.
This creates a stable skill gap and natural curriculum within self-play.

Architecture:
1. Phase 1 (Curriculum): Stages 0-19 with autopilot opponent, single-perspective training
2. Milestone (Stage 10): Save first checkpoint
3. Phase 2 (Curriculum): Stages 10-19 with autopilot opponent
4. Milestone (Stage 20): Save second checkpoint, START SELF-PLAY vs stage 10 checkpoint
5. Domination: When perf >= threshold, save new checkpoint and upgrade opponent

Key insight: We train the SAME policy from BOTH perspectives of the zero-sum game.
Each episode provides:
- "Here's what the winner did" → positive reward signal
- "Here's what the loser did" → negative reward signal

This doubles the learning signal and teaches the agent to both attack AND defend.

Usage:
    # Basic dual self-play training (starts self-play at stage 20)
    python pufferlib/ocean/dogfight/train_dual_selfplay.py

    # With wandb logging
    python pufferlib/ocean/dogfight/train_dual_selfplay.py --wandb --wandb-project df-dual

    # Start self-play immediately at stage 0 (for testing)
    python pufferlib/ocean/dogfight/train_dual_selfplay.py --selfplay-min-stage 0

    # Custom checkpoint queue settings
    python pufferlib/ocean/dogfight/train_dual_selfplay.py --checkpoint-lag 2 --perf-threshold 0.70

    # Verbose debug output
    python pufferlib/ocean/dogfight/train_dual_selfplay.py --debug

    # Evaluate player policy against opponent checkpoint with rendering
    python pufferlib/ocean/dogfight/train_dual_selfplay.py eval \
        --load-model-path experiments/model.pt \
        --opponent-checkpoint checkpoints/selfplay_xxx/checkpoint_stage20_step20000000.pt \
        --render-mode raylib

    # Or with wandb run ID for player model
    python pufferlib/ocean/dogfight/train_dual_selfplay.py eval \
        --load-id abc123 \
        --opponent-checkpoint path/to/opponent.pt
"""
import os
import sys
import copy
import time
import random
import argparse
import uuid
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

import pufferlib
import pufferlib.vector
import pufferlib.pytorch
from pufferlib import pufferl
from pufferlib.checkpoint_queue import CheckpointQueue

# Debug level: 0=off, 1=key events, 2=loop progress, 3=detailed
DEBUG_LEVEL = 0

def debug(level, msg):
    """Print debug message if level is enabled."""
    if DEBUG_LEVEL >= level:
        print(f'[DUAL-DEBUG-{level}] {msg}', flush=True)


# Configuration defaults
DEFAULT_OPPONENT_UPDATE_INTERVAL = 1_000_000  # Update opponent every 1M steps (legacy, unused with queue)
DEFAULT_SELFPLAY_MIN_STAGE = 20  # Only enable self-play after stage 20
DEFAULT_CHECKPOINT_LAG = 1  # Opponent is N checkpoints behind (1=2nd newest)
DEFAULT_PERF_THRESHOLD = 0.55  # Clean win rate gate for opponent advancement (AlphaGo Zero style)
DEFAULT_MIN_STEPS_BETWEEN_CHECKPOINTS = 2_000_000  # Minimum steps before saving new checkpoint
DEFAULT_MAX_CHECKPOINTS = 20  # Max selfplay checkpoints (milestones always kept)
DEFAULT_PAST_OPPONENT_PROB = 0.2  # 20% of games against past checkpoint pool (OpenAI Five: 20%)
DEFAULT_PFSP_EXPONENT = 2.0  # PFSP weighting exponent: (1-win_rate)^exp (OpenAI Five: squared)
DEFAULT_OPPONENT_RESAMPLE_INTERVAL = 1_000_000  # Re-roll opponent selection every N steps
DEFAULT_POOL_CHECKPOINT_INTERVAL = 5_000_000  # Save periodic checkpoint every N steps (ensures pool growth)
DEFAULT_OPPONENT_EPOCH_LENGTH = 5_000_000  # Steps per opponent in round-robin rotation
DEFAULT_MASTERY_STREAK = 2  # Full rotations of mastery before unlocking next rank
DEFAULT_DEBUG_TRIGGER_STEP = 500_000_000  # Start debug logging at 500M steps (0 = disabled)


class DualPerspectiveTrainer:
    """Trainer that collects experience from both player and opponent perspectives.

    During curriculum (stages 0-19):
    - Standard single-perspective PPO
    - Opponent is autopilot (handled by C code)

    During dual self-play (stage 20+):
    - Opponent is frozen copy of learner
    - Collect experience from BOTH perspectives
    - Train on combined experience (2x training signal)
    """

    def __init__(self, config, vecenv, learner_policy, logger=None,
                 opponent_update_interval=DEFAULT_OPPONENT_UPDATE_INTERVAL,
                 selfplay_min_stage=DEFAULT_SELFPLAY_MIN_STAGE,
                 checkpoint_lag=DEFAULT_CHECKPOINT_LAG,
                 perf_threshold=DEFAULT_PERF_THRESHOLD,
                 min_steps_between_checkpoints=DEFAULT_MIN_STEPS_BETWEEN_CHECKPOINTS,
                 max_checkpoints=DEFAULT_MAX_CHECKPOINTS,
                 past_opponent_prob=DEFAULT_PAST_OPPONENT_PROB,
                 pfsp_exponent=DEFAULT_PFSP_EXPONENT,
                 opponent_resample_interval=DEFAULT_OPPONENT_RESAMPLE_INTERVAL,
                 pool_checkpoint_interval=DEFAULT_POOL_CHECKPOINT_INTERVAL,
                 opponent_epoch_length=DEFAULT_OPPONENT_EPOCH_LENGTH,
                 mastery_streak=DEFAULT_MASTERY_STREAK,
                 checkpoint_dir=None,
                 run_id=None):
        # Store custom config
        self.opponent_update_interval = opponent_update_interval
        self.selfplay_min_stage = selfplay_min_stage
        self.checkpoint_lag = checkpoint_lag
        self.perf_threshold = perf_threshold
        self.min_steps_between_checkpoints = min_steps_between_checkpoints
        self.past_opponent_prob = past_opponent_prob
        self.pfsp_exponent = pfsp_exponent
        self.opponent_resample_interval = opponent_resample_interval
        self.pool_checkpoint_interval = pool_checkpoint_interval
        self.opponent_epoch_length = opponent_epoch_length
        self.mastery_streak_required = mastery_streak
        self.use_dual_selfplay = False
        self.last_opponent_update = 0

        # Create the standard PuffeRL trainer for the learner
        self.trainer = pufferl.PuffeRL(config, vecenv, learner_policy, logger)
        self.config = config
        self.vecenv = vecenv
        self.learner_policy = learner_policy

        # Generate run ID if not provided
        if run_id is None:
            run_id = str(uuid.uuid4())[:8]

        # Initialize checkpoint queue
        if checkpoint_dir is None:
            checkpoint_dir = f'checkpoints/selfplay_{run_id}'
        self.checkpoint_queue = CheckpointQueue(
            save_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints
        )

        # Track milestone saves and domination state
        self._saved_stage10 = False
        self._saved_stage20 = False
        self._current_opponent_path = None
        self._current_opponent_tag = None
        self.last_checkpoint_step = 0
        self._current_stage = 0
        self.last_resample_step = 0

        # PFSP: per-opponent win rate tracking (tag → exponential moving average)
        self.opponent_win_rates = {}
        self.pool_perf = 0.5  # EMA of win rate against pool opponents only

        # Ratchet state: progressive opponent unlocking with round-robin rotation
        self._unlocked_rank = 0              # Only increases (ratchet)
        self._current_rotation_idx = 0       # Which opponent in rotation
        self._rank_mastery_streak = 0        # Consecutive mastered rotations
        self._epoch_start_step = 0           # When current epoch started
        self._rotation_kills = 0.0           # Kills across current full rotation
        self._rotation_episodes = 0.0        # Episodes across current full rotation
        self._epoch_kills = 0.0              # Kills in current single-opponent epoch
        self._epoch_episodes = 0.0           # Episodes in current single-opponent epoch
        self._pool_perf_ema = 0.5            # Smoothed metric for wandb

        # Clean-win gate tracking (only accumulates during pool opponent epochs)
        self._gate_player_kills = 0.0
        self._gate_opp_kills = 0.0

        # Difficulty tracking metrics
        self._total_rank_ups = 0
        self._total_checkpoints_saved = 0
        self._learner_elo = 1500.0            # Online Elo (K=32 updates per epoch)
        self._checkpoint_elos = {}            # tag -> Elo rating

        # Stalemate detection and handicapping
        self.stalemate_perf_threshold = 0.3  # Low perf suggests stalemate
        self.stalemate_clean_threshold = 0.5  # Low clean fight rate suggests death spirals
        self.stalemate_counter = 0
        self.handicap_level = 0  # 0=none, 1=mild, 2=moderate, 3=severe
        self.opponent_handicap_controls = 1.0  # 1.0 = no handicap

        # Create frozen opponent policy (copy of learner)
        self.opponent_policy = None
        self._init_opponent_policy()

        # Get driver env for direct C access
        self.driver_env = vecenv.driver_env

        # Dual experience buffers (allocated lazily)
        self.opponent_obs = None
        self.opponent_actions = None
        self.opponent_logprobs = None
        self.opponent_values = None
        self.opponent_rewards = None
        self.opponent_terminals = None

        # Track opponent LSTM state if using RNN
        self.opponent_lstm_h = None
        self.opponent_lstm_c = None

        print(f'[DUAL-SELFPLAY] Initialized: min_stage={selfplay_min_stage}, '
              f'checkpoint_lag={checkpoint_lag}, perf_threshold={perf_threshold}')
        print(f'[DUAL-SELFPLAY] Ratchet: epoch_length={opponent_epoch_length}, '
              f'mastery_streak={mastery_streak}')
        print(f'[DUAL-SELFPLAY] Checkpoint dir: {checkpoint_dir}')

    def _init_opponent_policy(self):
        """Create frozen opponent policy as copy of learner."""
        device = self.config['device']

        # Deep copy the learner policy architecture
        self.opponent_policy = copy.deepcopy(self.learner_policy)
        self.opponent_policy = self.opponent_policy.to(device)

        # Copy current learner weights
        self.opponent_policy.load_state_dict(self.learner_policy.state_dict())

        # Freeze for inference only
        self.opponent_policy.eval()
        for p in self.opponent_policy.parameters():
            p.requires_grad = False

        print(f'[DUAL-SELFPLAY] Opponent policy initialized from learner')

    def _allocate_opponent_buffers(self):
        """Allocate experience buffers for opponent perspective."""
        if self.opponent_obs is not None:
            return  # Already allocated

        # Match learner buffer shapes
        device = self.config['device']
        segments = self.trainer.segments
        horizon = self.config['bptt_horizon']
        obs_space = self.vecenv.single_observation_space
        atn_space = self.vecenv.single_action_space

        self.opponent_obs = torch.zeros(segments, horizon, *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == 'cuda' and self.config.get('cpu_offload', False),
            device='cpu' if self.config.get('cpu_offload', False) else device)
        self.opponent_actions = torch.zeros(segments, horizon, *atn_space.shape, device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype])
        self.opponent_values = torch.zeros(segments, horizon, device=device)
        self.opponent_logprobs = torch.zeros(segments, horizon, device=device)
        self.opponent_rewards = torch.zeros(segments, horizon, device=device)
        self.opponent_terminals = torch.zeros(segments, horizon, device=device)

        print(f'[DUAL-SELFPLAY] Allocated opponent buffers: '
              f'obs={self.opponent_obs.shape}, actions={self.opponent_actions.shape}')
        # Verify allocation is zeros
        obs_range = (self.opponent_obs.min().item(), self.opponent_obs.max().item())
        debug(1, f'Opponent obs buffer after allocation: range={obs_range}')

    def _update_opponent(self):
        """Select opponent using 80/20 PFSP split.

        With probability (1 - past_opponent_prob): use current learner weights (self-play).
        With probability past_opponent_prob: sample from checkpoint pool using PFSP weighting,
        where harder opponents (lower win rate) are sampled more often.
        """
        pool_size = len(self.checkpoint_queue.checkpoints)

        if random.random() < self.past_opponent_prob and pool_size > 0:
            # Sample from pool using PFSP weighting
            opponent_path, opponent_tag = self._sample_pfsp_opponent()
            if opponent_path and opponent_path != self._current_opponent_path:
                self._load_opponent_from_checkpoint(opponent_path)
                self._current_opponent_path = opponent_path
                self._current_opponent_tag = opponent_tag
                self.last_opponent_update = self.trainer.global_step
                wr = self.opponent_win_rates.get(opponent_tag, 0.5)
                print(f'[PFSP] Sampled pool opponent: {opponent_tag} (win_rate={wr:.2f}, pool_size={pool_size})')
        else:
            # Use current learner weights (self-play against self)
            self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
            self._current_opponent_path = None
            self._current_opponent_tag = 'self'
            self.last_opponent_update = self.trainer.global_step
            debug(1, f'Using current learner weights as opponent (self-play)')

    def _sample_pfsp_opponent(self):
        """Sample opponent from checkpoint pool, weighted toward hard opponents.

        Uses Prioritized Fictitious Self-Play (PFSP) weighting:
            weight_i = (1 - win_rate_i) ^ pfsp_exponent

        Hard opponents (low win rate) get higher weight. This is the same approach
        used by OpenAI Five (exponent=2) and AlphaStar main agents (exponent=2).

        Returns:
            (path, tag) tuple, or (None, None) if pool empty.
        """
        entries = self.checkpoint_queue.checkpoints
        if not entries:
            return None, None

        # Compute PFSP weights
        weights = []
        for entry in entries:
            wr = self.opponent_win_rates.get(entry.tag, 0.5)  # Default 50% for unseen
            w = (1.0 - wr) ** self.pfsp_exponent
            weights.append(max(w, 1e-6))  # Prevent zero weights

        # Normalize and sample
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = random.choices(range(len(entries)), weights=probs, k=1)[0]
        return entries[idx].path, entries[idx].tag

    def _load_opponent_from_checkpoint(self, checkpoint_path: str):
        """Load opponent policy from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
        self.opponent_policy.load_state_dict(checkpoint['policy_state_dict'])

        # Get checkpoint info for logging
        tag = checkpoint.get('tag', 'unknown')
        step = checkpoint.get('step', 0)
        self._current_opponent_tag = tag
        print(f'[CHECKPOINT-QUEUE] Loaded opponent from {tag} (step {step}): {checkpoint_path}')

    def _check_milestone_save(self, current_stage: int):
        """Save checkpoints at stage 10 and stage 20 milestones."""
        if current_stage >= 10 and not self._saved_stage10:
            self.checkpoint_queue.save(
                self.learner_policy,
                self.trainer.global_step,
                current_stage,
                "stage10"
            )
            self._saved_stage10 = True
            self.last_checkpoint_step = self.trainer.global_step
            self._total_checkpoints_saved += 1
            self._checkpoint_elos["stage10"] = self._learner_elo
            print(f'[CHECKPOINT-QUEUE] Saved milestone: stage10 at step {self.trainer.global_step}')

        if current_stage >= 20 and not self._saved_stage20:
            self.checkpoint_queue.save(
                self.learner_policy,
                self.trainer.global_step,
                current_stage,
                "stage20"
            )
            self._saved_stage20 = True
            self.last_checkpoint_step = self.trainer.global_step
            self._total_checkpoints_saved += 1
            self._checkpoint_elos["stage20"] = self._learner_elo
            print(f'[CHECKPOINT-QUEUE] Saved milestone: stage20 at step {self.trainer.global_step}')

    def _update_opponent_win_rate(self, logs):
        """Update per-opponent win rate using exponential moving average.

        Called after each training epoch with logs containing perf metric.
        Uses EMA with decay 0.95 so recent performance matters more.
        Also updates aggregate pool_perf (only from pool opponent episodes).
        """
        if not self.use_dual_selfplay or not logs:
            return

        perf = logs.get('environment/perf', 0)
        tag = self._current_opponent_tag
        if tag and tag != 'self':
            # Update per-opponent win rate
            old_wr = self.opponent_win_rates.get(tag, 0.5)
            new_wr = 0.95 * old_wr + 0.05 * perf
            self.opponent_win_rates[tag] = new_wr
            # Update aggregate pool perf (only from pool fights)
            self.pool_perf = 0.95 * self.pool_perf + 0.05 * perf
            debug(2, f'Win rate update: {tag}: {old_wr:.3f} -> {new_wr:.3f} (perf={perf:.3f}, pool_perf={self.pool_perf:.3f})')

    def _check_resample_opponent(self):
        """Periodically re-roll opponent selection (the 80/20 dice).

        Without this, the opponent only changes on domination. For PFSP to work,
        we need to resample periodically so the agent faces varied opponents.
        """
        if not self.use_dual_selfplay:
            return

        steps_since_resample = self.trainer.global_step - self.last_resample_step
        if steps_since_resample >= self.opponent_resample_interval:
            self._update_opponent()
            self.last_resample_step = self.trainer.global_step

    def _check_periodic_checkpoint(self):
        """Save periodic checkpoints to ensure pool growth even without domination.

        OpenAI Five added checkpoints every 10 optimizer iterations. Without periodic saves,
        the pool never grows if the learner can't consistently beat old checkpoints, leading
        to no opponent diversity and catastrophic forgetting.
        """
        if not self.use_dual_selfplay:
            return

        steps_since_last = self.trainer.global_step - self.last_checkpoint_step
        if steps_since_last < self.pool_checkpoint_interval:
            return

        step_m = self.trainer.global_step // 1_000_000
        tag = f"periodic_{step_m}M"
        self.checkpoint_queue.save(
            self.learner_policy,
            self.trainer.global_step,
            self._current_stage,
            tag
        )
        self.last_checkpoint_step = self.trainer.global_step
        self._total_checkpoints_saved += 1
        self._checkpoint_elos[tag] = self._learner_elo

        # Initialize win rate at 0.5 (unknown)
        self.opponent_win_rates[tag] = 0.5

        pool_size = len(self.checkpoint_queue.checkpoints)
        print(f'[CHECKPOINT-QUEUE] Periodic save: {tag} (pool_size={pool_size})')

    def _get_sorted_opponents(self):
        """Return list of (path, tag) sorted by rank: milestones first, then periodic, then 'self'.

        Rank ordering:
        - stage10 (rank 0, weakest)
        - stage20 (rank 1)
        - periodic checkpoints sorted by step (rank 2+)
        - 'self' (highest rank, current learner weights)
        """
        entries = self.checkpoint_queue.checkpoints
        milestones = []
        periodics = []
        for e in entries:
            if e.tag == 'stage10':
                milestones.insert(0, (e.path, e.tag))  # stage10 first
            elif e.tag == 'stage20':
                milestones.append((e.path, e.tag))  # stage20 second
            else:
                periodics.append((e.path, e.tag, e.step))

        # Sort periodics by step (oldest = weakest)
        periodics.sort(key=lambda x: x[2])
        periodic_pairs = [(p, t) for p, t, _ in periodics]

        # Self is always last (highest rank)
        result = milestones + periodic_pairs + [(None, 'self')]
        return result

    def _load_opponent_for_rank(self, rank):
        """Load opponent for the given rank index in the sorted opponent list."""
        opponents = self._get_sorted_opponents()
        if rank >= len(opponents):
            rank = len(opponents) - 1
        path, tag = opponents[rank]
        if tag == 'self':
            self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
            self._current_opponent_path = None
            self._current_opponent_tag = 'self'
            debug(1, f'Loaded opponent rank {rank}: self (current learner weights)')
        else:
            self._load_opponent_from_checkpoint(path)
            self._current_opponent_path = path
            self._current_opponent_tag = tag
            debug(1, f'Loaded opponent rank {rank}: {tag}')

        print(f'[RATCHET] Epoch start: opponent={tag} (rank {rank}/{len(opponents)-1})')

    def _update_epoch_metrics(self, logs):
        """Accumulate kills and episodes from training logs into current epoch counters."""
        if not self.use_dual_selfplay or not logs:
            return
        perf = logs.get('environment/perf', 0)
        n = logs.get('environment/n', 0)
        if n > 0:
            self._epoch_kills += perf * n
            self._epoch_episodes += n

            # Gate tracking: only during pool opponent epochs (not self-play)
            if self._current_opponent_tag != 'self':
                sp_pk = logs.get('environment/sp_player_kills', 0)
                sp_ok = logs.get('environment/sp_opp_kills', 0)
                self._gate_player_kills += sp_pk * n
                self._gate_opp_kills += sp_ok * n

    def _update_elo(self, opponent_tag, learner_won):
        """Online Elo update (K=32) after each epoch against a pool opponent."""
        r_l = self._learner_elo
        r_o = self._checkpoint_elos.get(opponent_tag, 1500.0)
        e_l = 1.0 / (1.0 + 10.0 ** ((r_o - r_l) / 400.0))
        s_l = 1.0 if learner_won else 0.0
        K = 32
        self._learner_elo = r_l + K * (s_l - e_l)
        self._checkpoint_elos[opponent_tag] = r_o + K * ((1.0 - s_l) - (1.0 - e_l))

    def _check_epoch_boundary(self):
        """Check if current opponent epoch is complete, advance if so."""
        if not self.use_dual_selfplay:
            return
        steps_in_epoch = self.trainer.global_step - self._epoch_start_step
        if steps_in_epoch >= self.opponent_epoch_length:
            self._advance_to_next_epoch()

    def _advance_to_next_epoch(self):
        """End current epoch, accumulate into rotation, advance to next opponent."""
        # Accumulate current epoch into rotation totals
        self._rotation_kills += self._epoch_kills
        self._rotation_episodes += self._epoch_episodes

        # Log epoch perf
        epoch_perf = self._epoch_kills / max(self._epoch_episodes, 1)
        print(f'[RATCHET] Epoch done: opponent={self._current_opponent_tag}, '
              f'perf={epoch_perf:.3f} ({self._epoch_kills:.0f}/{self._epoch_episodes:.0f}), '
              f'rotation_idx={self._current_rotation_idx}')

        # Online Elo update (only against pool opponents, not self)
        if self._current_opponent_tag and self._current_opponent_tag != 'self':
            self._update_elo(self._current_opponent_tag, epoch_perf >= 0.5)

        # Reset epoch counters
        self._epoch_kills = 0.0
        self._epoch_episodes = 0.0

        # Advance rotation index
        opponents = self._get_sorted_opponents()
        num_active = min(self._unlocked_rank + 1, len(opponents))
        self._current_rotation_idx += 1

        if self._current_rotation_idx >= num_active:
            # Full rotation complete
            self._evaluate_rotation()
            self._current_rotation_idx = 0

        # Start next epoch
        self._epoch_start_step = self.trainer.global_step
        rank = self._current_rotation_idx  # rank = index in sorted list
        self._load_opponent_for_rank(rank)

    def _evaluate_rotation(self):
        """Evaluate gate after a full rotation through all unlocked opponents.

        Gate criteria: clean_win_rate >= perf_threshold (e.g. 0.55).
        Clean wins = gun kills only. Crashes and timeouts don't count.
        Inspired by AlphaGo Zero's 55% win-rate gate.
        """
        rotation_perf = self._rotation_kills / max(self._rotation_episodes, 1)
        opponents = self._get_sorted_opponents()
        num_active = min(self._unlocked_rank + 1, len(opponents))

        # Update EMA for wandb
        self._pool_perf_ema = 0.9 * self._pool_perf_ema + 0.1 * rotation_perf

        # Clean-win gate check using accumulated gate kills from pool opponent epochs
        total_kills = self._gate_player_kills + self._gate_opp_kills
        if total_kills >= 10:  # Minimum sample — need real data, not 1 lucky kill
            clean_win_rate = self._gate_player_kills / total_kills
            gate_passed = clean_win_rate >= self.perf_threshold
        else:
            clean_win_rate = 0.0
            gate_passed = False  # Not enough kills to judge — stay put

        print(f'[RATCHET] Rotation complete: perf={rotation_perf:.3f} '
              f'({self._rotation_kills:.0f}/{self._rotation_episodes:.0f}), '
              f'gate={clean_win_rate:.3f} ({self._gate_player_kills:.0f}pk/{total_kills:.0f}total, '
              f'need>={self.perf_threshold}), '
              f'unlocked_rank={self._unlocked_rank}, active={num_active}/{len(opponents)}, '
              f'streak={self._rank_mastery_streak}')

        if gate_passed:
            self._rank_mastery_streak += 1
            print(f'[RATCHET] Gate PASSED! streak={self._rank_mastery_streak}/{self.mastery_streak_required}')
            if self._rank_mastery_streak >= self.mastery_streak_required:
                # Unlock next opponent
                max_rank = len(opponents) - 1
                if self._unlocked_rank < max_rank:
                    self._unlocked_rank += 1
                    self._total_rank_ups += 1
                    new_tag = opponents[self._unlocked_rank][1] if self._unlocked_rank < len(opponents) else '?'
                    print(f'[RATCHET] RANK UP! unlocked_rank={self._unlocked_rank}, '
                          f'new opponent: {new_tag}')
                else:
                    print(f'[RATCHET] Already at max rank {self._unlocked_rank}')
                self._rank_mastery_streak = 0
        else:
            self._rank_mastery_streak = 0
            if total_kills < 10:
                print(f'[RATCHET] Gate INSUFFICIENT data ({total_kills:.0f} kills < 10 minimum), streak reset')
            else:
                print(f'[RATCHET] Gate FAILED (clean_win_rate={clean_win_rate:.3f} < {self.perf_threshold}), '
                      f'streak reset')

        # Reset rotation and gate counters
        self._rotation_kills = 0.0
        self._rotation_episodes = 0.0
        self._gate_player_kills = 0.0
        self._gate_opp_kills = 0.0

    def _check_stalemate(self, logs):
        """Check for stalemate (low perf, low clean fight rate) and apply handicaps.

        When both policies enter death spirals without kills:
        1. Detect via low perf AND low clean_fights rate
        2. Apply progressive handicaps to opponent
        3. If handicaps don't help, load older checkpoint
        """
        if not self.use_dual_selfplay:
            return

        if logs is None:
            return

        # Get metrics from logs
        perf = logs.get('environment/perf', 0)
        n = logs.get('environment/n', 1)
        clean_fights = logs.get('environment/clean_fights', 0)
        clean_rate = clean_fights / max(n, 1) if n > 0 else 0

        # Stalemate: low perf AND low clean fight rate (both spiral, no kills)
        is_stalemate = perf < self.stalemate_perf_threshold and clean_rate < self.stalemate_clean_threshold

        if is_stalemate:
            self.stalemate_counter += 1
            if self.stalemate_counter >= 5:  # 5 consecutive stalemate checks
                self._apply_handicap()
                self.stalemate_counter = 0
        else:
            # Decay counter when not in stalemate
            self.stalemate_counter = max(0, self.stalemate_counter - 1)

    def _apply_handicap(self):
        """Apply progressive handicap to opponent to break stalemate."""
        self.handicap_level = min(3, self.handicap_level + 1)

        if self.handicap_level == 1:
            # Level 1: Mild control reduction (90%)
            self.opponent_handicap_controls = 0.9
            print(f'[HANDICAP] Level 1: Opponent controls at 90%')
        elif self.handicap_level == 2:
            # Level 2: Moderate control reduction (80%) + older checkpoint
            self.opponent_handicap_controls = 0.8
            self.checkpoint_lag = min(self.checkpoint_lag + 1, 5)
            self._update_opponent()
            print(f'[HANDICAP] Level 2: Opponent controls at 80%, checkpoint_lag={self.checkpoint_lag}')
        elif self.handicap_level == 3:
            # Level 3: Severe control reduction (70%)
            self.opponent_handicap_controls = 0.7
            print(f'[HANDICAP] Level 3: Opponent controls at 70%')

    def _check_selfplay_transition(self, stats=None):
        """Check if we should transition to dual self-play mode and save milestones.

        Args:
            stats: Stats dict from trainer.stats (contains 'stage' from C logs)
        """
        # Get current stage from stats (populated by C code during evaluate)
        # Use avg_stage which is more reliable than individual episode stages
        # Use np.mean (not max) so self-play only activates when the COHORT has mastered
        # the curriculum, not when one lucky env races ahead while others are at stage 14
        if stats and 'avg_stage' in stats and len(stats['avg_stage']) > 0:
            current_stage = np.mean(stats['avg_stage'])
        elif stats and 'stage' in stats and len(stats['stage']) > 0:
            current_stage = np.mean(stats['stage'])
        else:
            # Last resort fallback
            current_stage = getattr(self.driver_env, '_current_stage', 0)
            if self.trainer.epoch % 100 == 0:
                print(f'[DUAL-SELFPLAY] WARNING: No stage in stats, using fallback stage={current_stage}')

        self._current_stage = current_stage

        # Check for milestone saves (stage 10, stage 20)
        self._check_milestone_save(int(current_stage))

        debug(1, f'_check_selfplay_transition: use_dual_selfplay={self.use_dual_selfplay}')
        if self.use_dual_selfplay:
            return  # Already in self-play mode

        # Trigger at 19.9+ to catch stage 20 reliably (avg_stage=20.0 when all episodes are stage 20)
        trigger_threshold = self.selfplay_min_stage - 0.1  # 20 - 0.1 = 19.9

        if current_stage >= trigger_threshold:
            print(f'[DUAL-SELFPLAY] Transitioning to dual self-play at stage {current_stage}', flush=True)
            self.use_dual_selfplay = True

            # Allocate opponent buffers
            self._allocate_opponent_buffers()

            # Enable opponent override in C code (activates self-play mode)
            # This must be done when transitioning to self-play, not at env init
            from pufferlib.ocean.dogfight import binding
            binding.vec_enable_opponent_override(self.driver_env.c_envs, 1)

            # Enable recovery hijacking now that we're in self-play
            # This breaks death spiral equilibrium by occasionally forcing opponent to recover
            binding.vec_set_selfplay_active(self.driver_env.c_envs, 1)
            print(f'[DUAL-SELFPLAY] Enabled recovery hijacking for death spiral prevention')

            # Signal workers to enable opponent override via shared memory flag
            # (workers check this flag in step() before using opponent actions)
            if hasattr(self.vecenv, 'buf') and 'selfplay_active' in self.vecenv.buf:
                self.vecenv.buf['selfplay_active'][0] = 1
                print(f'[DUAL-SELFPLAY] Set selfplay_active flag in shared memory')

            # Initialize ratchet rotation: start at rank 0 (stage10 = weakest)
            self._unlocked_rank = 0
            self._current_rotation_idx = 0
            self._epoch_start_step = self.trainer.global_step
            self._rank_mastery_streak = 0
            self._rotation_kills = 0.0
            self._rotation_episodes = 0.0
            self._epoch_kills = 0.0
            self._epoch_episodes = 0.0

            opponents = self._get_sorted_opponents()
            print(f'[RATCHET] Initialized rotation with {len(opponents)} opponents: '
                  f'{[t for _, t in opponents]}')
            self._load_opponent_for_rank(0)

    def evaluate(self):
        """Evaluate with dual experience collection in self-play mode."""
        # Note: selfplay transition check is done in train() BEFORE stats are cleared

        if not self.use_dual_selfplay:
            # Standard single-perspective evaluation
            return self.trainer.evaluate()

        # Dual self-play evaluation
        return self._evaluate_dual()

    def _evaluate_dual(self):
        """Collect experience from both player AND opponent perspectives."""
        debug(1, f'_evaluate_dual starting: segments={self.trainer.segments}')

        profile = self.trainer.profile
        epoch = self.trainer.epoch
        profile('eval', epoch)
        profile('eval_misc', epoch, nest=True)

        config = self.config
        device = config['device']

        # Import binding for C-level access
        from pufferlib.ocean.dogfight import binding

        # Reset LSTM states if using RNN
        if config['use_rnn']:
            for k in self.trainer.lstm_h:
                self.trainer.lstm_h[k].zero_()
                self.trainer.lstm_c[k].zero_()
            # Initialize opponent LSTM if needed
            if self.opponent_lstm_h is None:
                n = self.vecenv.agents_per_batch
                h = self.learner_policy.hidden_size
                total_agents = self.trainer.total_agents
                self.opponent_lstm_h = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
                self.opponent_lstm_c = {i*n: torch.zeros(n, h, device=device) for i in range(total_agents//n)}
            for k in self.opponent_lstm_h:
                self.opponent_lstm_h[k].zero_()
                self.opponent_lstm_c[k].zero_()

        self.trainer.full_rows = 0
        loop_count = 0
        while self.trainer.full_rows < self.trainer.segments:
            loop_count += 1
            if loop_count % 100 == 1:
                debug(2, f'eval loop {loop_count}: full_rows={self.trainer.full_rows}/{self.trainer.segments}')

            profile('env', epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile('eval_misc', epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            debug(3, f'recv: o.shape={o.shape}, env_id={env_id}')

            done_mask = d + t
            self.trainer.global_step += int(mask.sum())

            # Update C-side global_step for shaping reward decay
            # This ensures accurate timestep tracking during self-play
            binding.vec_set_global_step(self.driver_env.c_envs, self.trainer.global_step)

            profile('eval_copy', epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)
            r = torch.as_tensor(r).to(device)
            d = torch.as_tensor(d).to(device)

            # Get opponent observations from shared memory buffers (Multiprocessing)
            # or via C binding (Serial). C code writes to buffers during c_step().
            if hasattr(self.vecenv, 'buf') and 'opponent_observations' in self.vecenv.buf:
                # Multiprocessing: read from shared memory buffer
                o_opponent_all = self.vecenv.buf['opponent_observations']
                debug(3, f'opponent obs from buf: shape={o_opponent_all.shape}')
                # buf shape is (num_workers, agents_per_worker, *obs_shape)
                # w_slice from recv() gives us the right worker indices
                o_opponent = torch.as_tensor(o_opponent_all[self.vecenv.w_slice].reshape(-1, *self.vecenv.single_observation_space.shape)).to(device)
            else:
                # Serial: use C binding directly
                o_opponent_all = binding.vec_get_opponent_observations(self.driver_env.c_envs)
                debug(3, f'opponent obs from binding: all.shape={o_opponent_all.shape}, slicing with env_id={env_id}')
                o_opponent = torch.as_tensor(o_opponent_all[env_id]).to(device)

            # Handle NaN observations (can occur at episode boundaries)
            # Replace NaN with zeros - these will get masked out anyway
            nan_count = np.isnan(o_opponent_all[env_id]).sum() if isinstance(o_opponent_all, np.ndarray) else 0
            if nan_count > 0:
                debug(2, f'NaN in opponent obs: {nan_count} values')
            if torch.isnan(o_opponent).any():
                o_opponent = torch.nan_to_num(o_opponent, nan=0.0)
                # Verify cleaning worked
                if torch.isnan(o_opponent).any():
                    debug(1, f'ERROR: NaN still in o_opponent after nan_to_num!')
                else:
                    debug(3, f'NaN cleaned successfully')

            profile('eval_forward', epoch)
            with torch.no_grad(), self.trainer.amp_context:
                # Learner forward pass
                state_p = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config['use_rnn']:
                    state_p['lstm_h'] = self.trainer.lstm_h[env_id.start]
                    state_p['lstm_c'] = self.trainer.lstm_c[env_id.start]

                logits_p, value_p = self.trainer.policy.forward_eval(o_device, state_p)
                action_p, logprob_p, _ = pufferlib.pytorch.sample_logits(logits_p)
                r_clamped = torch.clamp(r, -1, 1)

                # Opponent forward pass (no gradients, frozen)
                state_o = dict(
                    reward=-r,  # Opponent gets negative reward
                    done=d,
                    env_id=env_id,
                    mask=mask,
                    lstm_h=None,
                    lstm_c=None,
                )

                if config['use_rnn']:
                    state_o['lstm_h'] = self.opponent_lstm_h[env_id.start]
                    state_o['lstm_c'] = self.opponent_lstm_c[env_id.start]

                logits_o, value_o = self.opponent_policy.forward_eval(o_opponent, state_o)
                action_o, logprob_o, _ = pufferlib.pytorch.sample_logits(logits_o)

                # GUIDED CLIMB OVERRIDE: Check if any envs have teachable climb active
                # When active, override opponent actions with climb control BEFORE recording
                # This creates training data showing "climb after merge = good strategy"
                guided_state = binding.vec_get_guided_climb_state(self.driver_env.c_envs)
                # guided_state shape: (num_envs, 3) = [active, ticks_remaining, elevator]
                climb_active_mask = guided_state[:, 0] > 0.5  # Boolean mask
                if climb_active_mask.any():
                    # Override actions for envs with guided climb active
                    # Actions: [throttle, elevator, aileron, rudder, trigger]
                    # Climb control: full throttle, pull up (elevator), wings level, no rudder, no trigger
                    climb_actions = torch.zeros_like(action_o)
                    climb_actions[:, 0] = 1.0   # Full throttle for energy
                    climb_actions[:, 1] = torch.tensor(guided_state[:, 2], device=device)  # Elevator from C
                    climb_actions[:, 2] = 0.0   # Wings level
                    climb_actions[:, 3] = 0.0   # No rudder
                    climb_actions[:, 4] = 0.0   # No trigger

                    # Apply override only to climbing envs (use mask to select)
                    mask_tensor = torch.tensor(climb_active_mask, device=device).unsqueeze(1)
                    action_o = torch.where(mask_tensor, climb_actions, action_o)

                    # Tick the climb counter (decrement remaining ticks)
                    binding.vec_tick_guided_climb(self.driver_env.c_envs)

                    debug(2, f'Guided climb active for {climb_active_mask.sum()} envs')

            debug(3, f'actions: player={action_p.shape}, opponent={action_o.shape}')

            profile('eval_copy', epoch)
            with torch.no_grad():
                # Update LSTM states
                if config['use_rnn']:
                    self.trainer.lstm_h[env_id.start] = state_p['lstm_h']
                    self.trainer.lstm_c[env_id.start] = state_p['lstm_c']
                    self.opponent_lstm_h[env_id.start] = state_o['lstm_h']
                    self.opponent_lstm_c[env_id.start] = state_o['lstm_c']

                # Fast path for fully vectorized envs
                l = self.trainer.ep_lengths[env_id.start].item()
                batch_rows = slice(self.trainer.ep_indices[env_id.start].item(),
                                   1+self.trainer.ep_indices[env_id.stop - 1].item())

                # Store PLAYER experience
                if config.get('cpu_offload', False):
                    self.trainer.observations[batch_rows, l] = o
                else:
                    self.trainer.observations[batch_rows, l] = o_device

                self.trainer.actions[batch_rows, l] = action_p
                self.trainer.logprobs[batch_rows, l] = logprob_p
                self.trainer.rewards[batch_rows, l] = r_clamped
                self.trainer.terminals[batch_rows, l] = d.float()
                self.trainer.values[batch_rows, l] = value_p.flatten()

                # Store OPPONENT experience (rewards are NEGATIVE of player)
                if config.get('cpu_offload', False):
                    self.opponent_obs[batch_rows, l] = o_opponent.cpu()
                else:
                    self.opponent_obs[batch_rows, l] = o_opponent

                self.opponent_actions[batch_rows, l] = action_o
                self.opponent_logprobs[batch_rows, l] = logprob_o
                self.opponent_rewards[batch_rows, l] = -r_clamped  # ZERO-SUM
                self.opponent_terminals[batch_rows, l] = d.float()
                self.opponent_values[batch_rows, l] = value_o.flatten()

                # Handle episode boundaries
                self.trainer.ep_lengths[env_id] += 1
                if l+1 >= config['bptt_horizon']:
                    num_full = env_id.stop - env_id.start
                    self.trainer.ep_indices[env_id] = self.trainer.free_idx + torch.arange(
                        num_full, device=config['device']).int()
                    self.trainer.ep_lengths[env_id] = 0
                    self.trainer.free_idx += num_full
                    self.trainer.full_rows += num_full

                # Prepare actions for env
                action_p_np = action_p.cpu().numpy()
                action_o_np = action_o.cpu().numpy()

                if isinstance(logits_p, torch.distributions.Normal):
                    action_p_np = np.clip(action_p_np,
                                          self.vecenv.action_space.low,
                                          self.vecenv.action_space.high)
                    action_o_np = np.clip(action_o_np,
                                          self.vecenv.action_space.low,
                                          self.vecenv.action_space.high)

                # Apply handicap to opponent controls (elevator, aileron, rudder)
                # Throttle (index 0) and trigger (index 4) are not reduced
                if self.opponent_handicap_controls < 1.0:
                    action_o_np = action_o_np.copy()
                    action_o_np[:, 1:4] *= self.opponent_handicap_controls

            profile('eval_misc', epoch)
            # Process info
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.trainer.stats[k].extend(v)
                    else:
                        self.trainer.stats[k].append(v)

            # Set opponent actions: write to shared memory (Multiprocessing) or C binding (Serial)
            profile('env', epoch)
            if hasattr(self.vecenv, 'buf') and 'opponent_actions' in self.vecenv.buf:
                # Multiprocessing: write to shared memory buffer
                # Workers will read this during their step() call
                opp_act_buf = self.vecenv.buf['opponent_actions']
                opp_act_buf[self.vecenv.w_slice] = action_o_np.reshape(opp_act_buf[self.vecenv.w_slice].shape)
            else:
                # Serial: set directly via C binding
                binding.vec_set_opponent_actions(self.driver_env.c_envs, action_o_np)

            # Send player actions
            self.vecenv.send(action_p_np)

        profile('eval_misc', epoch)
        self.trainer.free_idx = self.trainer.total_agents
        self.trainer.ep_indices = torch.arange(self.trainer.total_agents,
                                               device=device, dtype=torch.int32)
        self.trainer.ep_lengths.zero_()
        profile.end()
        # Verify opponent obs after evaluate
        obs_range = (self.opponent_obs.min().item(), self.opponent_obs.max().item())
        debug(1, f'Opponent obs after _evaluate_dual: range={obs_range}')

        debug(1, f'_evaluate_dual done: loops={loop_count}, global_step={self.trainer.global_step}')
        return self.trainer.stats

    def train(self):
        """Train on combined experience in self-play mode."""
        # Inject strength into stats BEFORE train() so it appears as environment/strength
        # in W&B during both curriculum and self-play phases
        strength = float(self._unlocked_rank) / float(max(self.checkpoint_queue.max_checkpoints, 1))
        self.trainer.stats['strength'] = [strength]

        if not self.use_dual_selfplay:
            # Standard single-perspective training
            # Check for selfplay transition BEFORE train() clears stats
            self._check_selfplay_transition(self.trainer.stats)
            logs = self.trainer.train()
            return logs

        # Dual self-play training
        logs = self._train_dual()

        # Ratchet: track epoch metrics and check for epoch/rotation boundaries
        self._update_epoch_metrics(logs)
        self._check_epoch_boundary()

        # Periodic checkpoint save (ensures pool growth)
        self._check_periodic_checkpoint()

        # Check for stalemate -> apply handicaps to break death spiral equilibrium
        self._check_stalemate(logs)

        return logs

    def _train_dual(self):
        """Train on combined player + opponent experience."""
        debug(1, f'_train_dual starting: epoch={self.trainer.epoch}')

        profile = self.trainer.profile
        epoch = self.trainer.epoch
        profile('train', epoch)
        profile('train_misc', epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        device = config['device']

        # Combine player and opponent experience
        # Shape: [2*segments, horizon, ...]
        combined_obs = torch.cat([self.trainer.observations, self.opponent_obs], dim=0)
        combined_actions = torch.cat([self.trainer.actions, self.opponent_actions], dim=0)
        combined_logprobs = torch.cat([self.trainer.logprobs, self.opponent_logprobs], dim=0)
        combined_values = torch.cat([self.trainer.values, self.opponent_values], dim=0)
        combined_rewards = torch.cat([self.trainer.rewards, self.opponent_rewards], dim=0)
        combined_terminals = torch.cat([self.trainer.terminals, self.opponent_terminals], dim=0)
        combined_ratio = torch.cat([self.trainer.ratio, torch.ones_like(self.trainer.ratio)], dim=0)

        # Handle NaN and extreme values in observations
        if torch.isnan(combined_obs).any():
            nan_before = torch.isnan(combined_obs).sum().item()
            combined_obs = torch.nan_to_num(combined_obs, nan=0.0)
            debug(1, f'Cleaned NaN in combined_obs: {nan_before} values')

        # Check for extreme values (observations should be in [-1, 1] range)
        obs_min, obs_max = combined_obs.min().item(), combined_obs.max().item()
        if obs_min < -100 or obs_max > 100:
            debug(1, f'WARNING: Extreme obs values! range=[{obs_min:.2e}, {obs_max:.2e}]')
            # Check player vs opponent buffers
            p_min, p_max = self.trainer.observations.min().item(), self.trainer.observations.max().item()
            o_min, o_max = self.opponent_obs.min().item(), self.opponent_obs.max().item()
            debug(1, f'  player_obs range: [{p_min:.2e}, {p_max:.2e}]')
            debug(1, f'  opponent_obs range: [{o_min:.2e}, {o_max:.2e}]')
            # Clamp to sane range
            combined_obs = torch.clamp(combined_obs, -10, 10)
            debug(1, f'  Clamped to [-10, 10]')

        total_segments = combined_values.shape[0]

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*epoch/self.trainer.total_epochs

        # Reset ratio for combined experience
        self.trainer.ratio[:] = 1

        debug(2, f'train: total_minibatches={self.trainer.total_minibatches}, combined shape={combined_values.shape}')

        # Check for NaN in combined buffers
        obs_nan = torch.isnan(combined_obs).sum().item()
        if obs_nan > 0:
            debug(1, f'WARNING: NaN in combined_obs: {obs_nan} values')
            # Find which buffer has NaN
            p_nan = torch.isnan(self.trainer.observations).sum().item()
            o_nan = torch.isnan(self.opponent_obs).sum().item()
            debug(1, f'  player_obs NaN: {p_nan}, opponent_obs NaN: {o_nan}')

        # Check for NaN in rewards (opposite signs could cause issues)
        reward_nan = torch.isnan(combined_rewards).sum().item()
        if reward_nan > 0:
            debug(1, f'WARNING: NaN in combined_rewards: {reward_nan} values')

        for mb in range(self.trainer.total_minibatches):
            if mb % 10 == 0:
                debug(3, f'train minibatch {mb}/{self.trainer.total_minibatches}')
            profile('train_misc', epoch)
            self.trainer.amp_context.__enter__()

            shape = combined_values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = pufferl.compute_puff_advantage(combined_values, combined_rewards,
                combined_terminals, combined_ratio, advantages, config['gamma'],
                config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)

            # Sample from combined segments (2x as many)
            idx = torch.multinomial(prio_probs, self.trainer.minibatch_segments)
            mb_prio = (total_segments*prio_probs[idx, None])**-anneal_beta

            profile('train_copy', epoch)
            mb_obs = combined_obs[idx]
            mb_actions = combined_actions[idx]
            mb_logprobs = combined_logprobs[idx]
            mb_rewards = combined_rewards[idx]
            mb_terminals = combined_terminals[idx]
            mb_ratio = combined_ratio[idx]
            mb_values = combined_values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile('train_forward', epoch)
            if not config['use_rnn']:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            # Forward pass through LEARNER policy (gets gradients from BOTH perspectives)
            if torch.isnan(mb_obs).any():
                debug(1, f'ERROR: NaN in mb_obs before forward pass! mb={mb}')
                debug(1, f'  mb_obs shape: {mb_obs.shape}, NaN count: {torch.isnan(mb_obs).sum().item()}')

            # Check policy weights for NaN before forward
            for name, p in self.trainer.policy.named_parameters():
                if torch.isnan(p).any():
                    debug(1, f'ERROR: NaN in policy param {name} BEFORE forward!')
                    break

            try:
                logits, newvalue = self.trainer.policy(mb_obs, state)
            except ValueError as e:
                debug(1, f'ERROR in forward pass: {e}')
                # Check all the inputs
                debug(1, f'  mb_obs: shape={mb_obs.shape}, nan={torch.isnan(mb_obs).sum().item()}, inf={torch.isinf(mb_obs).sum().item()}')
                debug(1, f'  mb_obs range: [{mb_obs.min().item():.4f}, {mb_obs.max().item():.4f}]')
                # Check hidden state from LSTM
                if 'hidden' in state:
                    h = state['hidden']
                    debug(1, f'  hidden: nan={torch.isnan(h).sum().item()}, inf={torch.isinf(h).sum().item()}')
                # Re-raise to stop training
                raise

            # Check if logits contain NaN
            if isinstance(logits, torch.distributions.Normal):
                if torch.isnan(logits.loc).any() or torch.isnan(logits.scale).any():
                    debug(1, f'ERROR: NaN in logits! loc_nan={torch.isnan(logits.loc).sum().item()}, scale_nan={torch.isnan(logits.scale).sum().item()}')
            elif torch.isnan(logits).any():
                debug(1, f'ERROR: NaN in logits tensor!')
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile('train_misc', epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            combined_ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            # Weight advantages by priority and normalize
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # PPO losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            self.trainer.amp_context.__enter__()  # TODO: AMP needs debugging

            # Update values for combined buffer (only player portion used for priority)
            combined_values[idx] = newvalue.detach().float()

            # Logging
            profile('train_misc', epoch)
            losses['policy_loss'] += pg_loss.item() / self.trainer.total_minibatches
            losses['value_loss'] += v_loss.item() / self.trainer.total_minibatches
            losses['entropy'] += entropy_loss.item() / self.trainer.total_minibatches
            losses['old_approx_kl'] += old_approx_kl.item() / self.trainer.total_minibatches
            losses['approx_kl'] += approx_kl.item() / self.trainer.total_minibatches
            losses['clipfrac'] += clipfrac.item() / self.trainer.total_minibatches
            losses['importance'] += ratio.mean().item() / self.trainer.total_minibatches

            # Learn on accumulated minibatches
            profile('learn', epoch)
            loss.backward()
            if (mb + 1) % self.trainer.accumulate_minibatches == 0:
                # Check for NaN in gradients before stepping
                grad_nan = False
                for name, p in self.trainer.policy.named_parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        debug(1, f'WARNING: NaN grad in {name}')
                        grad_nan = True
                if grad_nan:
                    debug(1, f'NaN gradient detected at mb {mb}, skipping optimizer step')
                    self.trainer.optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(self.trainer.policy.parameters(), config['max_grad_norm'])
                self.trainer.optimizer.step()
                self.trainer.optimizer.zero_grad()

                # Check for NaN in weights after stepping
                for name, p in self.trainer.policy.named_parameters():
                    if torch.isnan(p).any():
                        debug(1, f'WARNING: NaN weight in {name} after step')
                        break

        # Update learning rate scheduler
        profile('train_misc', epoch)
        if config['anneal_lr']:
            self.trainer.scheduler.step()

        y_pred = combined_values.flatten()
        y_true = advantages.flatten() + combined_values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        # Add dual self-play specific metrics
        losses['dual_selfplay'] = 1.0 if self.use_dual_selfplay else 0.0
        losses['pool_win_rate'] = self._pool_perf_ema
        losses['pool_size'] = float(len(self.checkpoint_queue.checkpoints))
        losses['vs_pool'] = 0.0 if self._current_opponent_tag == 'self' else 1.0
        losses['unlocked_rank'] = float(self._unlocked_rank)
        losses['pool_perf_ema'] = self._pool_perf_ema
        losses['epoch_type'] = 1.0 if self._current_opponent_tag == 'self' else 0.0
        losses['opponent_rank'] = float(self._current_rotation_idx)

        # Difficulty tracking metrics (7 new wandb metrics)
        # 1. Training steps behind current opponent (0 when fighting self)
        opponent_step_lag = 0.0
        if self._current_opponent_tag and self._current_opponent_tag != 'self':
            for entry in self.checkpoint_queue.checkpoints:
                if entry.tag == self._current_opponent_tag:
                    opponent_step_lag = float(self.trainer.global_step - entry.step)
                    break
        losses['opponent_step_lag'] = opponent_step_lag

        # 2. Cumulative rank-up counter (monotonically increasing)
        losses['total_rank_ups'] = float(self._total_rank_ups)

        # 3. Progress toward next rank-up (0.0 to 1.0, resets on rank-up or failure)
        losses['mastery_progress'] = float(self._rank_mastery_streak) / max(float(self.mastery_streak_required), 1.0)

        # 4. Kill rate against current opponent this epoch
        losses['epoch_perf'] = self._epoch_kills / max(self._epoch_episodes, 1.0)

        # 5. Normalized opponent difficulty (0.0=weakest, 1.0=self)
        opponents = self._get_sorted_opponents()
        total_opponents = len(opponents)
        losses['opponent_relative_strength'] = float(self._current_rotation_idx) / max(float(total_opponents - 1), 1.0)

        # 6. Total checkpoints ever saved (pool generation count)
        losses['pool_generation'] = float(self._total_checkpoints_saved)

        # 7. Online Elo rating (tracks skill progression within a run)
        losses['learner_elo'] = self._learner_elo

        # 8. Clean-win gate metrics
        total_gate_kills = self._gate_player_kills + self._gate_opp_kills
        losses['gate_clean_win_rate'] = self._gate_player_kills / max(total_gate_kills, 1)
        losses['gate_player_kills'] = self._gate_player_kills
        losses['gate_opp_kills'] = self._gate_opp_kills

        # 9. Strength: normalized progress through opponent ladder (sweep metric)
        strength = float(self._unlocked_rank) / float(max(self.checkpoint_queue.max_checkpoints, 1))
        losses['strength'] = strength

        # Also inject strength into stats so it appears as environment/strength for sweep metric
        self.trainer.stats['strength'] = [strength]

        profile.end()
        logs = None
        self.trainer.epoch += 1
        self.trainer.losses = losses
        done_training = self.trainer.global_step >= config['total_timesteps']
        if done_training or self.trainer.global_step == 0 or time.time() > self.trainer.last_log_time + 0.25:
            logs = self.trainer.mean_and_log()
            self.trainer.print_dashboard()
            self.trainer.stats = defaultdict(list)
            self.trainer.last_log_time = time.time()
            self.trainer.last_log_step = self.trainer.global_step
            profile.clear()

        if self.trainer.epoch % config['checkpoint_interval'] == 0 or done_training:
            self.trainer.save_checkpoint()
            self.trainer.msg = f'Checkpoint saved at update {self.trainer.epoch}'

        return logs

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def epoch(self):
        return self.trainer.epoch

    def close(self):
        return self.trainer.close()


def train_dual(env_name='puffer_dogfight', args=None, should_stop_early=None):
    """Train with dual self-play. Returns all_logs like pufferl.train().

    This is the core training function extracted for sweep support.
    """
    global DEBUG_LEVEL

    args = args or pufferl.load_config(env_name)

    # NOTE: Dual self-play works with Multiprocessing backend!
    # Opponent observations and rewards are computed in C during c_step()
    # and written to shared memory buffers, enabling parallel workers.
    backend = args['vec'].get('backend', 'Multiprocessing')
    print(f'[DUAL-SELFPLAY] Using {backend} backend with C-level opponent buffers')

    # Create environment using standard pufferl flow
    vecenv = pufferl.load_env(env_name, args)

    # Create policy
    policy = pufferl.load_policy(args, vecenv, env_name)

    # Create logger if requested
    logger = None
    run_id = None
    if args['neptune']:
        logger = pufferl.NeptuneLogger(args)
    elif args['wandb']:
        logger = pufferl.WandbLogger(args)
        # Use wandb run ID for checkpoint directory
        if hasattr(logger, 'run') and logger.run:
            run_id = logger.run.id

    # Get selfplay params from [selfplay] INI section first, then fall back to top-level args
    selfplay_args = args.get('selfplay', {})
    selfplay_min_stage = args.get('selfplay_min_stage', DEFAULT_SELFPLAY_MIN_STAGE)
    checkpoint_lag = args.get('checkpoint_lag', DEFAULT_CHECKPOINT_LAG)
    perf_threshold = selfplay_args.get('perf_threshold',
                     args.get('perf_threshold', DEFAULT_PERF_THRESHOLD))
    min_steps_between_checkpoints = args.get('min_steps_between_checkpoints', DEFAULT_MIN_STEPS_BETWEEN_CHECKPOINTS)
    max_checkpoints = args.get('max_checkpoints', DEFAULT_MAX_CHECKPOINTS)
    checkpoint_dir = args.get('checkpoint_dir', None)
    opponent_update_interval = args.get('opponent_update_interval', DEFAULT_OPPONENT_UPDATE_INTERVAL)
    past_opponent_prob = selfplay_args.get('past_opponent_prob', DEFAULT_PAST_OPPONENT_PROB)
    pfsp_exponent = selfplay_args.get('pfsp_exponent', DEFAULT_PFSP_EXPONENT)
    opponent_resample_interval = int(selfplay_args.get('opponent_resample_interval', DEFAULT_OPPONENT_RESAMPLE_INTERVAL))
    pool_checkpoint_interval = int(selfplay_args.get('pool_checkpoint_interval', DEFAULT_POOL_CHECKPOINT_INTERVAL))
    opponent_epoch_length = int(selfplay_args.get('opponent_epoch_length', DEFAULT_OPPONENT_EPOCH_LENGTH))
    mastery_streak = int(selfplay_args.get('mastery_streak', DEFAULT_MASTERY_STREAK))

    # Create dual-perspective trainer with checkpoint queue
    train_config = {**args['train'], 'env': env_name}
    trainer = DualPerspectiveTrainer(
        train_config, vecenv, policy, logger,
        opponent_update_interval=opponent_update_interval,
        selfplay_min_stage=selfplay_min_stage,
        checkpoint_lag=checkpoint_lag,
        perf_threshold=perf_threshold,
        min_steps_between_checkpoints=min_steps_between_checkpoints,
        max_checkpoints=max_checkpoints,
        past_opponent_prob=past_opponent_prob,
        pfsp_exponent=pfsp_exponent,
        opponent_resample_interval=opponent_resample_interval,
        pool_checkpoint_interval=pool_checkpoint_interval,
        opponent_epoch_length=opponent_epoch_length,
        mastery_streak=mastery_streak,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id
    )

    print(f'[DUAL-SELFPLAY] Starting training with ratchet opponent curriculum')
    print(f'[DUAL-SELFPLAY] Min stage for self-play: {selfplay_min_stage}')
    print(f'[DUAL-SELFPLAY] Perf threshold: {perf_threshold} (mastery threshold per rotation)')
    print(f'[DUAL-SELFPLAY] Ratchet: epoch_length={opponent_epoch_length}, mastery_streak={mastery_streak}')
    print(f'[DUAL-SELFPLAY] Pool checkpoint interval: {pool_checkpoint_interval} (periodic saves for pool growth)')

    total_timesteps = train_config['total_timesteps']
    all_logs = []

    # Training loop
    while trainer.global_step < total_timesteps:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        trainer.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = trainer.train()

        if logs is not None:
            # Only collect after 20% warmup (matching pufferl.train)
            if trainer.global_step > 0.20 * total_timesteps:
                all_logs.append(logs)

            if should_stop_early is not None and should_stop_early(logs):
                model_path = trainer.close()
                if logger:
                    logger.close(model_path)
                return all_logs, model_path

        # Log dual self-play status periodically
        if trainer.epoch % 100 == 0 and trainer.epoch > 0:
            mode = "DUAL" if trainer.use_dual_selfplay else "CURRICULUM"
            queue_len = len(trainer.checkpoint_queue)
            opp_tag = trainer._current_opponent_tag or "none"
            print(f'[DUAL-SELFPLAY] Mode: {mode}, Steps: {trainer.global_step}, '
                  f'Queue: {queue_len} checkpoints, Opponent: {opp_tag}, '
                  f'Rank: {trainer._unlocked_rank}, PerfEMA: {trainer._pool_perf_ema:.3f}')

    # Cleanup
    model_path = trainer.close()
    if logger:
        logger.close(model_path)

    print(f'[DUAL-SELFPLAY] Training complete')
    return all_logs, model_path


def sweep_dual(env_name='puffer_dogfight', args=None):
    """Sweep hyperparameters for dual self-play training.

    Mirrors pufferl.sweep() but uses train_dual() instead of train().
    """
    args = args or pufferl.load_config(env_name)

    if not args['wandb'] and not args['neptune']:
        raise pufferlib.APIUsageError('Sweeps require either wandb or neptune')

    method = args['sweep'].pop('method')

    project = args.get('wandb_project', args.get('neptune_project', 'sweep'))
    args['sweep'].setdefault('state_file', f'{project}_sweep.json')
    args['sweep'].setdefault('override_file', f'{project}_override.json')

    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f'Invalid sweep method {method}. See pufferlib.sweep')

    sweep = sweep_cls(args['sweep'])
    points_per_run = args['sweep']['downsample']
    target_key = f'environment/{args["sweep"]["metric"]}'

    def stop_if_loss_nan(logs):
        return any("losses/" in k and np.isnan(v) for k, v in logs.items())

    for i in range(args['max_runs']):
        seed = time.time_ns() & 0xFFFFFFFF
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sweep.suggest(args)
        all_logs, model_path = train_dual(env_name, args=args, should_stop_early=stop_if_loss_nan)

        # Post-training Elo evaluation
        elo_eval_cfg = args.get('elo_eval', {})
        if elo_eval_cfg.get('enabled', False) and model_path:
            from pufferlib.ocean.dogfight.elo_eval import run_benchmark_eval, load_manifest
            ref_dir = elo_eval_cfg.get('reference_dir', 'pufferlib/ocean/dogfight/reference_opponents')
            manifest = load_manifest(ref_dir)
            if manifest:
                try:
                    result = run_benchmark_eval(
                        model_path=model_path,
                        reference_opponents=manifest['opponents'],
                        games_per_matchup=int(elo_eval_cfg.get('games_per_matchup', 20)),
                        obs_scheme=args['env'].get('obs_scheme', 0),
                        device=args['train']['device'],
                    )
                    # Inject Elo into logs so Protein can optimize it
                    if all_logs:
                        all_logs[-1]['environment/elo'] = result['elo'] / 1000.0
                    print(f'[ELO-EVAL] Rating: {result["elo"]:.0f} ({result["eval_time_seconds"]:.1f}s)')
                except Exception as e:
                    print(f'[ELO-EVAL] Failed: {e}')

        all_logs = [e for e in all_logs if target_key in e]

        if not all_logs:
            sweep.observe(args, 0, 0, is_failure=True)
            continue

        total_timesteps = args['train']['total_timesteps']

        scores = pufferl.downsample([log[target_key] for log in all_logs], points_per_run)
        costs = pufferl.downsample([log['uptime'] for log in all_logs], points_per_run)
        timesteps = pufferl.downsample([log['agent_steps'] for log in all_logs], points_per_run)

        if len(timesteps) > 0 and timesteps[-1] < 0.7 * total_timesteps:
            s = scores.pop()
            c = costs.pop()
            args['train']['total_timesteps'] = timesteps.pop()
            sweep.observe(args, s, c, is_failure=True)

        for score, cost, timestep in zip(scores, costs, timesteps):
            args['train']['total_timesteps'] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args['train']['total_timesteps'] = total_timesteps


def eval_selfplay(env_name, args, player_path, opponent_path, load_id=None):
    """Evaluate player policy against opponent checkpoint with rendering.

    This is like pufferl.eval() but with dual policy inference:
    - Player uses the loaded model (from player_path or load_id)
    - Opponent uses a checkpoint from opponent_path

    Args:
        env_name: Environment name ('puffer_dogfight')
        args: Config args dict
        player_path: Path to player model weights (or None if using load_id)
        opponent_path: Path to opponent checkpoint file
        load_id: Optional wandb/neptune run ID to load player model from
    """
    from pufferlib.ocean.dogfight import binding

    # Force Serial backend with single env for eval
    backend = args['vec'].get('backend', 'Serial')
    if backend != 'PufferEnv':
        backend = 'Serial'
    args['vec'] = dict(backend=backend, num_envs=1)
    args['env']['num_envs'] = 1  # Also set internal agent count to 1

    # Enable eval spawn mode: truly random positions, angles, alternating advantages
    args['env']['curriculum_randomize'] = 1
    print(f'[EVAL-SELFPLAY] Enabled curriculum_randomize for varied spawn positions')

    # Create environment
    vecenv = pufferl.load_env(env_name, args)

    # Load player policy
    # Create policy first, then load weights manually to handle our checkpoint format
    device = args['train']['device']
    player_policy = pufferl.load_policy(args, vecenv, env_name)

    if player_path:
        checkpoint = torch.load(player_path, map_location=device)
        if 'policy_state_dict' in checkpoint:
            # Our checkpoint format (from CheckpointQueue)
            player_policy.load_state_dict(checkpoint['policy_state_dict'])
            tag = checkpoint.get('tag', 'unknown')
            step = checkpoint.get('step', 0)
            print(f'[EVAL-SELFPLAY] Loaded player from {tag} (step {step}): {player_path}')
        else:
            # Raw state dict format
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            player_policy.load_state_dict(state_dict)
            print(f'[EVAL-SELFPLAY] Loaded player from: {player_path}')

    player_policy.eval()

    # Create opponent policy (same architecture, different weights)
    opponent_policy = copy.deepcopy(player_policy)

    # Load opponent weights from checkpoint
    checkpoint = torch.load(opponent_path, map_location=device)
    if 'policy_state_dict' in checkpoint:
        # Our checkpoint format
        opponent_policy.load_state_dict(checkpoint['policy_state_dict'])
        tag = checkpoint.get('tag', 'unknown')
        step = checkpoint.get('step', 0)
        print(f'[EVAL-SELFPLAY] Loaded opponent from {tag} (step {step}): {opponent_path}')
    else:
        # Raw state dict format
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        opponent_policy.load_state_dict(state_dict)
        print(f'[EVAL-SELFPLAY] Loaded opponent from: {opponent_path}')

    opponent_policy.eval()
    for p in opponent_policy.parameters():
        p.requires_grad = False

    # Get driver env for rendering and C-level access
    driver = vecenv.driver_env
    num_agents = vecenv.num_envs  # Actual batch size, not observation dimension

    # Enable opponent override in C code (so autopilot doesn't control opponent)
    binding.vec_enable_opponent_override(driver.c_envs, 1)
    print(f'[EVAL-SELFPLAY] Enabled opponent override (neural network opponent)')

    # Initialize LSTM states if using RNN
    state_p = {}
    state_o = {}
    if args['train']['use_rnn']:
        state_p = dict(
            lstm_h=torch.zeros(num_agents, player_policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, player_policy.hidden_size, device=device),
        )
        state_o = dict(
            lstm_h=torch.zeros(num_agents, opponent_policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, opponent_policy.hidden_size, device=device),
        )

    # Reset environment with time-based seed for variety
    seed = int(time.time_ns() % 2**31)
    ob, info = vecenv.reset(seed=seed)

    frames = []
    episode_count = 0
    step_count = 0

    # Get config values with sensible defaults
    save_frames = args.get('save_frames', 0)
    fps = args.get('fps', 15)
    gif_path = args.get('gif_path', 'selfplay_eval.gif')

    print(f'[EVAL-SELFPLAY] Starting evaluation loop (press ESC to exit)')
    print(f'[EVAL-SELFPLAY] Player: {player_path or load_id}')
    print(f'[EVAL-SELFPLAY] Opponent: {opponent_path}')

    while True:
        # Render
        render = driver.render()
        if len(frames) < save_frames:
            frames.append(render)

        # Handle different render modes
        if driver.render_mode == 'ansi':
            print('\033[0;0H' + render + '\n')
            time.sleep(1 / fps)
        elif driver.render_mode == 'rgb_array':
            # raylib handles its own display, but we need to throttle
            time.sleep(1 / fps)

        # Get player observation
        ob_tensor = torch.as_tensor(ob).to(device)

        # Get opponent observation from C - slice to match actual num_agents
        # (C returns observations for all env slots, but we only use first num_agents)
        ob_opponent_np = binding.vec_get_opponent_observations(driver.c_envs)
        ob_opponent_np = ob_opponent_np[:num_agents]  # Only take first num_agents rows
        ob_opponent = torch.as_tensor(ob_opponent_np).to(device)

        # Handle NaN in opponent observations (can occur at boundaries)
        if torch.isnan(ob_opponent).any():
            ob_opponent = torch.nan_to_num(ob_opponent, nan=0.0)

        with torch.no_grad():
            # Player forward pass
            logits_p, value_p = player_policy.forward_eval(ob_tensor, state_p)
            action_p, logprob_p, _ = pufferlib.pytorch.sample_logits(logits_p)
            action_p_np = action_p.cpu().numpy().reshape(vecenv.action_space.shape)

            # Opponent forward pass
            logits_o, value_o = opponent_policy.forward_eval(ob_opponent, state_o)
            action_o, logprob_o, _ = pufferlib.pytorch.sample_logits(logits_o)
            # Keep 2D shape (num_agents, action_dim) - don't reshape to lose batch dimension!
            action_o_np = action_o.cpu().numpy().astype(np.float32)

        # Clip actions for continuous action space
        if isinstance(logits_p, torch.distributions.Normal):
            action_p_np = np.clip(action_p_np, vecenv.action_space.low, vecenv.action_space.high)
            action_o_np = np.clip(action_o_np, vecenv.action_space.low, vecenv.action_space.high)

        # Debug: print BOTH observations and actions every 60 steps
        if step_count % 60 == 0:
            obs_p = ob.flatten() if hasattr(ob, 'flatten') else ob[0]
            obs_o = ob_opponent_np[0] if len(ob_opponent_np.shape) > 1 else ob_opponent_np
            act_p = action_p_np.flatten()
            act_o = action_o_np.flatten()
            # Obs scheme 0: [fwd_spd, sideslip, climb, roll_r, pitch_r, yaw_r,
            #                aoa, altitude, energy, tgt_az, tgt_el, range, closure,
            #                E_adv, aspect, timer]
            print(f'[DEBUG] step={step_count}')
            print(f'  PLAYER OBS: tgt_az={obs_p[9]:.2f} tgt_el={obs_p[10]:.2f} range={obs_p[11]:.2f} closure={obs_p[12]:.2f} aspect={obs_p[14]:.2f}')
            print(f'  PLAYER ACT: throttle={act_p[0]:.2f} elev={act_p[1]:.2f} ail={act_p[2]:.2f} rud={act_p[3]:.2f} trig={act_p[4]:.2f}')
            print(f'  OPPON  OBS: tgt_az={obs_o[9]:.2f} tgt_el={obs_o[10]:.2f} range={obs_o[11]:.2f} closure={obs_o[12]:.2f} aspect={obs_o[14]:.2f}')
            print(f'  OPPON  ACT: throttle={act_o[0]:.2f} elev={act_o[1]:.2f} ail={act_o[2]:.2f} rud={act_o[3]:.2f} trig={act_o[4]:.2f}')

        # Set opponent actions in C (before stepping) - already correct 2D shape (num_agents, 5)
        binding.vec_set_opponent_actions(driver.c_envs, action_o_np)

        # Step environment with player action
        ob, reward, terminated, truncated, info = vecenv.step(action_p_np)
        step_count += 1

        # Check for episode end
        if terminated.any() or truncated.any():
            episode_count += 1
            if episode_count % 10 == 0:
                print(f'[EVAL-SELFPLAY] Episode {episode_count} completed (step {step_count})')

            # Reset with time-based seed for variety in next episode
            seed = int(time.time_ns() % 2**31)
            ob, info = vecenv.reset(seed=seed)

            # Reset LSTM states for fresh episode
            if args['train']['use_rnn']:
                state_p = dict(
                    lstm_h=torch.zeros(num_agents, player_policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, player_policy.hidden_size, device=device),
                )
                state_o = dict(
                    lstm_h=torch.zeros(num_agents, opponent_policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, opponent_policy.hidden_size, device=device),
                )

        # Save frames to gif if requested
        if len(frames) > 0 and len(frames) == save_frames:
            import imageio
            imageio.mimsave(gif_path, frames, fps=fps, loop=0)
            print(f'[EVAL-SELFPLAY] Saved {len(frames)} frames to {gif_path}')
            frames = []  # Reset to allow more recording


def main():
    global DEBUG_LEVEL
    env_name = 'puffer_dogfight'

    # Check for 'eval' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        sys.argv.pop(1)  # Remove 'eval' from args

        # Parse eval-specific args
        opponent_checkpoint = None
        player_path = None
        load_id = None
        no_rnn = False

        new_argv = [sys.argv[0]]
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]

            # --no-rnn flag
            if arg == '--no-rnn':
                no_rnn = True
                i += 1
                continue

            # --opponent-checkpoint <path>
            if arg == '--opponent-checkpoint':
                if i + 1 < len(sys.argv):
                    opponent_checkpoint = sys.argv[i + 1]
                    i += 2
                    continue
            elif arg.startswith('--opponent-checkpoint='):
                opponent_checkpoint = arg.split('=', 1)[1]
                i += 1
                continue

            # --load-model-path <path> (for player)
            if arg == '--load-model-path':
                if i + 1 < len(sys.argv):
                    player_path = sys.argv[i + 1]
                    i += 2
                    continue
            elif arg.startswith('--load-model-path='):
                player_path = arg.split('=', 1)[1]
                i += 1
                continue

            # --load-id <id> (for player, from wandb/neptune)
            if arg == '--load-id':
                if i + 1 < len(sys.argv):
                    load_id = sys.argv[i + 1]
                    i += 2
                    continue
            elif arg.startswith('--load-id='):
                load_id = arg.split('=', 1)[1]
                i += 1
                continue

            new_argv.append(arg)
            i += 1

        sys.argv = new_argv

        # Validate required args
        if opponent_checkpoint is None:
            print('Error: --opponent-checkpoint is required for eval mode')
            print('Usage: python train_dual_selfplay.py eval --opponent-checkpoint <path> [--load-model-path <path> | --load-id <id>]')
            sys.exit(1)

        if player_path is None and load_id is None:
            print('Error: Either --load-model-path or --load-id is required for eval mode')
            print('Usage: python train_dual_selfplay.py eval --opponent-checkpoint <path> [--load-model-path <path> | --load-id <id>]')
            sys.exit(1)

        # Load config
        args = pufferl.load_config(env_name)

        # Override RNN if requested
        if no_rnn:
            args['rnn_name'] = None
            args['train']['use_rnn'] = False
            print(f'[EVAL-SELFPLAY] RNN disabled (--no-rnn flag)')

        print(f'[EVAL-SELFPLAY] Starting eval mode')
        print(f'[EVAL-SELFPLAY] Player model: {player_path or load_id}')
        print(f'[EVAL-SELFPLAY] Opponent checkpoint: {opponent_checkpoint}')

        # Run eval
        eval_selfplay(env_name, args, player_path, opponent_checkpoint, load_id)
        return

    # Check for 'sweep' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == 'sweep':
        sys.argv.pop(1)  # Remove 'sweep' from args
        args = pufferl.load_config(env_name)
        sweep_dual(env_name, args)
        return

    # Check for 'train' subcommand (optional, for consistency)
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)  # Remove 'train' from args

    # Extract custom args before pufferl parses
    opponent_update_interval = DEFAULT_OPPONENT_UPDATE_INTERVAL
    selfplay_min_stage = DEFAULT_SELFPLAY_MIN_STAGE
    checkpoint_lag = DEFAULT_CHECKPOINT_LAG
    perf_threshold = DEFAULT_PERF_THRESHOLD
    min_steps_between_checkpoints = DEFAULT_MIN_STEPS_BETWEEN_CHECKPOINTS
    max_checkpoints = DEFAULT_MAX_CHECKPOINTS
    checkpoint_dir = None

    def parse_int_arg(args, i, arg_name):
        """Parse --arg value or --arg=value format, return (value, new_i) or None."""
        if args[i] == f'--{arg_name}':
            if i + 1 < len(args):
                return int(args[i + 1]), i + 2
        elif args[i].startswith(f'--{arg_name}='):
            return int(args[i].split('=', 1)[1]), i + 1
        return None

    def parse_float_arg(args, i, arg_name):
        """Parse --arg value or --arg=value format, return (value, new_i) or None."""
        if args[i] == f'--{arg_name}':
            if i + 1 < len(args):
                return float(args[i + 1]), i + 2
        elif args[i].startswith(f'--{arg_name}='):
            return float(args[i].split('=', 1)[1]), i + 1
        return None

    def parse_str_arg(args, i, arg_name):
        """Parse --arg value or --arg=value format, return (value, new_i) or None."""
        if args[i] == f'--{arg_name}':
            if i + 1 < len(args):
                return args[i + 1], i + 2
        elif args[i].startswith(f'--{arg_name}='):
            return args[i].split('=', 1)[1], i + 1
        return None

    new_argv = []
    i = 0
    while i < len(sys.argv):
        # Legacy: opponent-update-interval (kept for compatibility)
        result = parse_int_arg(sys.argv, i, 'opponent-update-interval')
        if result:
            opponent_update_interval = result[0]
            i = result[1]
            continue

        result = parse_int_arg(sys.argv, i, 'selfplay-min-stage')
        if result:
            selfplay_min_stage = result[0]
            i = result[1]
            continue

        # New checkpoint queue args
        result = parse_int_arg(sys.argv, i, 'checkpoint-lag')
        if result:
            checkpoint_lag = result[0]
            i = result[1]
            continue

        result = parse_float_arg(sys.argv, i, 'perf-threshold')
        if result:
            perf_threshold = result[0]
            i = result[1]
            continue

        result = parse_int_arg(sys.argv, i, 'min-checkpoint-gap')
        if result:
            min_steps_between_checkpoints = result[0]
            i = result[1]
            continue

        result = parse_int_arg(sys.argv, i, 'max-checkpoints')
        if result:
            max_checkpoints = result[0]
            i = result[1]
            continue

        result = parse_str_arg(sys.argv, i, 'checkpoint-dir')
        if result:
            checkpoint_dir = result[0]
            i = result[1]
            continue

        if sys.argv[i] == '--debug':
            DEBUG_LEVEL = 2
            i += 1
            continue
        elif sys.argv[i] == '--debug-verbose':
            DEBUG_LEVEL = 3
            i += 1
            continue

        new_argv.append(sys.argv[i])
        i += 1
    sys.argv = new_argv

    # Parse debug-trigger-step after other args (stored in global for training loop)
    global _debug_trigger_step_override
    _debug_trigger_step_override = None
    for i, arg in enumerate(sys.argv):
        if arg == '--debug-trigger-step' and i + 1 < len(sys.argv):
            _debug_trigger_step_override = int(sys.argv[i + 1])
            sys.argv = sys.argv[:i] + sys.argv[i+2:]
            break
        elif arg.startswith('--debug-trigger-step='):
            _debug_trigger_step_override = int(arg.split('=', 1)[1])
            sys.argv = sys.argv[:i] + sys.argv[i+1:]
            break

    # Load standard dogfight config
    args = pufferl.load_config(env_name)

    # Pass selfplay params to args for train_dual() to use
    args['selfplay_min_stage'] = selfplay_min_stage
    args['checkpoint_lag'] = checkpoint_lag
    args['perf_threshold'] = perf_threshold
    args['min_steps_between_checkpoints'] = min_steps_between_checkpoints
    args['max_checkpoints'] = max_checkpoints
    args['checkpoint_dir'] = checkpoint_dir
    args['opponent_update_interval'] = opponent_update_interval

    # Run training
    train_dual(env_name, args=args)


if __name__ == '__main__':
    main()
