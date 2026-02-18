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
from pufferlib.ocean.dogfight.dogfight_log import init_log, log

# Debug level: 0=off, 1=key events, 2=loop progress, 3=detailed
DEBUG_LEVEL = 0

def debug(level, msg):
    """Log debug message if level is enabled."""
    if DEBUG_LEVEL >= level:
        log(f'[SELFPLAY] debug={level} {msg}')


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

# Vertical merge curriculum: forced vertical spawn scenarios during self-play
DEFAULT_VERTICAL_PROB = 0.10                # 10% of self-play episodes use forced vertical spawns
DEFAULT_VERTICAL_RAMP_STEPS = 50_000_000    # Steps to progress through all 5 levels (0-4)


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
                 run_id=None,
                 skip_curriculum=False,
                 vertical_prob=DEFAULT_VERTICAL_PROB,
                 league_opponent_pool=None,
                 antiforgetting_pool=None,
                 self_play_prob=None,
                 league_prob=None,
                 antiforgetting_prob=None):
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
        self._selfplay_confirm_count = 0
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
        self._gate_clean_fights = 0.0
        self._gate_total_episodes = 0.0

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

        # League pools: external opponent lists for league training mode
        self.league_opponent_pool = league_opponent_pool  # List of (path, tag)
        self.antiforgetting_pool = antiforgetting_pool  # List of (path, tag)
        # Opponent split probabilities (None = use defaults based on mode)
        self._self_play_prob = self_play_prob
        self._league_prob = league_prob
        self._antiforgetting_prob = antiforgetting_prob

        # Vertical merge curriculum: tracks when self-play started for level progression
        self._vertical_prob = vertical_prob
        self._vertical_selfplay_start_step = None
        self._vertical_last_log_step = 0

        log(f'[SELFPLAY] event=init min_stage={selfplay_min_stage} checkpoint_lag={checkpoint_lag} perf_threshold={perf_threshold}')
        log(f'[SELFPLAY] ratchet epoch_length={opponent_epoch_length} mastery_streak={mastery_streak}')
        log(f'[SELFPLAY] checkpoint_dir={checkpoint_dir}')

        # skip_curriculum: immediately activate self-play mode (for league training)
        if skip_curriculum:
            self._activate_selfplay_immediately()

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

        log(f'[SELFPLAY] event=opponent_init source=learner')

    def _activate_selfplay_immediately(self):
        """Activate self-play mode immediately, skipping curriculum.

        Used by league training to start in self-play mode from tick 0.
        Replicates the activation logic from _check_selfplay_transition().
        """
        log(f'[SELFPLAY] event=skip_curriculum activating_immediately=true')
        self.use_dual_selfplay = True

        self._allocate_opponent_buffers()

        from pufferlib.ocean.dogfight import binding
        binding.vec_enable_opponent_override(self.driver_env.c_envs, 1)
        binding.vec_set_selfplay_active(self.driver_env.c_envs, 1)

        # Signal workers via shared memory
        if hasattr(self.vecenv, 'buf') and 'selfplay_active' in self.vecenv.buf:
            self.vecenv.buf['selfplay_active'][0] = 1

        # Initialize ratchet at rank 0
        self._unlocked_rank = 0
        self._current_rotation_idx = 0
        self._epoch_start_step = self.trainer.global_step
        self._rank_mastery_streak = 0
        self._rotation_kills = 0.0
        self._rotation_episodes = 0.0
        self._epoch_kills = 0.0
        self._epoch_episodes = 0.0
        self._gate_clean_fights = 0.0
        self._gate_total_episodes = 0.0

        # If league pools are set, start with self-play opponent
        self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
        self._current_opponent_path = None
        self._current_opponent_tag = 'self'

        # Start vertical merge curriculum timer
        self._vertical_selfplay_start_step = self.trainer.global_step

        log(f'[SELFPLAY] event=activated stage=0')

    def _update_vertical_curriculum(self):
        """Progress vertical merge curriculum based on steps since self-play activation."""
        if self._vertical_selfplay_start_step is None:
            return

        steps_in_selfplay = self.trainer.global_step - self._vertical_selfplay_start_step
        ramp_steps = DEFAULT_VERTICAL_RAMP_STEPS

        # Level progresses 0→4 over ramp_steps, then stays at 4
        level = min(4, int(steps_in_selfplay / (ramp_steps / 5)))

        # Constant probability from config (no decay)
        prob = self._vertical_prob

        from pufferlib.ocean.dogfight import binding
        binding.vec_set_vertical_curriculum(self.driver_env.c_envs, prob, level)

        # Log periodically (every ~5M steps)
        if self.trainer.global_step - self._vertical_last_log_step >= 5_000_000:
            self._vertical_last_log_step = self.trainer.global_step
            log(f'[VERTICAL] level={level} prob={prob:.3f} steps_in_selfplay={steps_in_selfplay}')

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

        log(f'[SELFPLAY] event=buffers_allocated obs={self.opponent_obs.shape} actions={self.opponent_actions.shape}')
        # Verify allocation is zeros
        obs_range = (self.opponent_obs.min().item(), self.opponent_obs.max().item())
        debug(1, f'Opponent obs buffer after allocation: range={obs_range}')

    def _update_opponent(self):
        """Select opponent using split probabilities.

        When league_opponent_pool is set (league mode), uses 3-way split:
          self_play_prob (default 0.35): current learner weights
          league_prob (default 0.50): PFSP from league pool
          antiforgetting_prob (default 0.15): uniform from antiforgetting pool

        Otherwise uses standard 80/20 split:
          (1 - past_opponent_prob): self-play
          past_opponent_prob: PFSP from checkpoint pool
        """
        if self.league_opponent_pool is not None:
            self._update_opponent_league()
            return

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
                log(f'[SELFPLAY] event=sample_opponent tag={opponent_tag} wr={wr:.2f} pool_size={pool_size}')
        else:
            # Use current learner weights (self-play against self)
            self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
            self._current_opponent_path = None
            self._current_opponent_tag = 'self'
            self.last_opponent_update = self.trainer.global_step
            debug(1, f'Using current learner weights as opponent (self-play)')

    def _update_opponent_league(self):
        """Select opponent using 3-way league split (35/50/15)."""
        sp_prob = self._self_play_prob if self._self_play_prob is not None else 0.35
        lg_prob = self._league_prob if self._league_prob is not None else 0.50
        # antiforgetting_prob is the remainder

        r = random.random()

        if r < sp_prob:
            # Self-play: use current learner weights
            self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
            self._current_opponent_path = None
            self._current_opponent_tag = 'self'
            self.last_opponent_update = self.trainer.global_step
            log(f'[SELFPLAY] event=sample_self step={self.trainer.global_step}')

        elif r < sp_prob + lg_prob and self.league_opponent_pool:
            # League PFSP: sample from league pool weighted by difficulty
            path, tag = self._sample_league_pfsp()
            if path:
                self._load_opponent_from_path(path)
                self._current_opponent_path = path
                self._current_opponent_tag = tag
                self.last_opponent_update = self.trainer.global_step
                wr = self.opponent_win_rates.get(tag, 0.5)
                log(f'[SELFPLAY] event=sample_league tag={tag} wr={wr:.2f}')
            else:
                # Fallback to self-play
                self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
                self._current_opponent_path = None
                self._current_opponent_tag = 'self'
                self.last_opponent_update = self.trainer.global_step

        elif self.antiforgetting_pool:
            # Anti-forgetting: uniform sample from anchor pool
            path, tag = random.choice(self.antiforgetting_pool)
            self._load_opponent_from_path(path)
            self._current_opponent_path = path
            self._current_opponent_tag = tag
            self.last_opponent_update = self.trainer.global_step
            log(f'[SELFPLAY] event=antiforgetting tag={tag}')

        else:
            # No pools available, fallback to self-play
            self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
            self._current_opponent_path = None
            self._current_opponent_tag = 'self'
            self.last_opponent_update = self.trainer.global_step

    def _sample_league_pfsp(self):
        """Sample from league opponent pool using PFSP weighting."""
        if not self.league_opponent_pool:
            return None, None

        weights = []
        for path, tag in self.league_opponent_pool:
            wr = self.opponent_win_rates.get(tag, 0.45)
            w = (1.0 - wr) ** self.pfsp_exponent
            weights.append(max(w, 1e-6))

        total = sum(weights)
        probs = [w / total for w in weights]
        idx = random.choices(range(len(self.league_opponent_pool)), weights=probs, k=1)[0]
        chosen_tag = self.league_opponent_pool[idx][1]
        weight_strs = ' '.join(f'{self.league_opponent_pool[i][1]}={probs[i]:.2f}' for i in range(len(probs)))
        log(f'[SELFPLAY] event=pfsp_sample chosen={chosen_tag} weights={weight_strs}')
        return self.league_opponent_pool[idx]

    def _load_opponent_from_path(self, path):
        """Load opponent policy from a raw .pt file (PuffeRL, CheckpointQueue, or full state dict)."""
        state_dict = torch.load(path, map_location=self.config['device'], weights_only=True)

        if isinstance(state_dict, dict) and 'policy_state_dict' in state_dict:
            # CheckpointQueue format
            self.opponent_policy.load_state_dict(state_dict['policy_state_dict'])
        else:
            # Try direct load first (handles full LSTMWrapper state dicts)
            try:
                self.opponent_policy.load_state_dict(state_dict)
            except RuntimeError:
                # Fallback: clean keys for PuffeRL format
                cleaned = {}
                for k, v in state_dict.items():
                    new_k = k.replace('module.', '').replace('policy.', '')
                    cleaned[new_k] = v
                self.opponent_policy.load_state_dict(cleaned)

        self.opponent_policy.eval()
        for p in self.opponent_policy.parameters():
            p.requires_grad = False

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
            wr = self.opponent_win_rates.get(entry.tag, 0.45)  # Default: slightly favor unknowns
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
        log(f'[CHECKPOINT] event=loaded tag={tag} step={step} path={checkpoint_path}')

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
            log(f'[CHECKPOINT] event=milestone tag=stage10 step={self.trainer.global_step}')

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
            log(f'[CHECKPOINT] event=milestone tag=stage20 step={self.trainer.global_step}')

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
            old_wr = self.opponent_win_rates.get(tag, 0.45)
            new_wr = 0.80 * old_wr + 0.20 * perf
            self.opponent_win_rates[tag] = new_wr
            # Update aggregate pool perf (only from pool fights)
            self.pool_perf = 0.80 * self.pool_perf + 0.20 * perf
            log(f'[SELFPLAY] event=wr_update tag={tag} old={old_wr:.3f} new={new_wr:.3f} perf={perf:.3f} pool_perf={self.pool_perf:.3f}')

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

        # Initialize win rate at 0.45 (slightly favor unknowns)
        self.opponent_win_rates[tag] = 0.45

        pool_size = len(self.checkpoint_queue.checkpoints)
        log(f'[CHECKPOINT] event=periodic tag={tag} pool_size={pool_size}')

    def _get_sorted_opponents(self):
        """Return list of (path, tag) sorted by rank: milestones first, then periodic, then league, then 'self'.

        Rank ordering:
        - stage10 (rank 0, weakest)
        - stage20 (rank 1)
        - periodic checkpoints sorted by step (rank 2+)
        - league pool opponents (higher rank than self-play checkpoints)
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

        # Include league pool opponents in rotation (after self-play checkpoints, before self)
        # Tag with 'league:' prefix so _load_opponent_for_rank uses the right loader
        league_entries = []
        if self.league_opponent_pool:
            for path, tag in self.league_opponent_pool:
                league_entries.append((path, f'league:{tag}'))

        # Self is always last (highest rank)
        result = milestones + periodic_pairs + league_entries + [(None, 'self')]
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
        elif tag.startswith('league:'):
            # League pool entries use raw .pt files, not CheckpointQueue format
            self._load_opponent_from_path(path)
            self._current_opponent_path = path
            self._current_opponent_tag = tag
            debug(1, f'Loaded opponent rank {rank}: {tag}')
        else:
            self._load_opponent_from_checkpoint(path)
            self._current_opponent_path = path
            self._current_opponent_tag = tag
            debug(1, f'Loaded opponent rank {rank}: {tag}')

        log(f'[RATCHET] event=epoch_start opponent={tag} rank={rank}/{len(opponents)-1}')

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
        log(f'[RATCHET] event=epoch_done opponent={self._current_opponent_tag} perf={epoch_perf:.3f} kills={self._epoch_kills:.0f} episodes={self._epoch_episodes:.0f} rotation_idx={self._current_rotation_idx}')

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
        CLEAN_FIGHT_GATE = 0.80
        total_kills = self._gate_player_kills + self._gate_opp_kills
        clean_fight_rate = self._gate_clean_fights / max(self._gate_total_episodes, 1)
        if total_kills >= 10:  # Minimum sample — need real data, not 1 lucky kill
            clean_win_rate = self._gate_player_kills / total_kills
            # Both conditions must pass:
            # 1. Win 55% of gun kills
            # 2. At least 80% of episodes must be clean (kills or timeouts, not crashes)
            gate_passed = (clean_win_rate >= self.perf_threshold
                           and clean_fight_rate >= CLEAN_FIGHT_GATE)
        else:
            clean_win_rate = 0.0
            gate_passed = False  # Not enough kills to judge — stay put

        log(f'[RATCHET] event=rotation_done perf={rotation_perf:.3f} kills={self._rotation_kills:.0f} episodes={self._rotation_episodes:.0f} gate={clean_win_rate:.3f} pk={self._gate_player_kills:.0f} total_kills={total_kills:.0f} threshold={self.perf_threshold} clean_fight_rate={clean_fight_rate:.3f} cfr_threshold={CLEAN_FIGHT_GATE} unlocked_rank={self._unlocked_rank} active={num_active}/{len(opponents)} streak={self._rank_mastery_streak}')

        if gate_passed:
            self._rank_mastery_streak += 1
            log(f'[RATCHET] event=gate_pass streak={self._rank_mastery_streak}/{self.mastery_streak_required}')
            if self._rank_mastery_streak >= self.mastery_streak_required:
                # Unlock next opponent
                max_rank = len(opponents) - 1
                if self._unlocked_rank < max_rank:
                    self._unlocked_rank += 1
                    self._total_rank_ups += 1
                    new_tag = opponents[self._unlocked_rank][1] if self._unlocked_rank < len(opponents) else '?'
                    log(f'[RATCHET] event=rank_up rank={self._unlocked_rank} new_opponent={new_tag} pool_size={len(opponents)}')
                else:
                    log(f'[RATCHET] event=max_rank rank={self._unlocked_rank}')
                self._rank_mastery_streak = 0
        else:
            self._rank_mastery_streak = 0
            if total_kills < 10:
                log(f'[RATCHET] event=gate_fail reason=insufficient_data kills={total_kills:.0f} minimum=10 streak_reset=true')
            else:
                log(f'[RATCHET] event=gate_fail wr={clean_win_rate:.3f} threshold={self.perf_threshold} clean_fight_rate={clean_fight_rate:.3f} cfr_threshold={CLEAN_FIGHT_GATE} streak_reset=true')

        # Reset rotation and gate counters
        self._rotation_kills = 0.0
        self._rotation_episodes = 0.0
        self._gate_player_kills = 0.0
        self._gate_opp_kills = 0.0
        self._gate_clean_fights = 0.0
        self._gate_total_episodes = 0.0

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
            log(f'[SELFPLAY] event=handicap level=1 controls=90%')
        elif self.handicap_level == 2:
            # Level 2: Moderate control reduction (80%) + older checkpoint
            self.opponent_handicap_controls = 0.8
            self.checkpoint_lag = min(self.checkpoint_lag + 1, 5)
            self._update_opponent()
            log(f'[SELFPLAY] event=handicap level=2 controls=80% checkpoint_lag={self.checkpoint_lag}')
        elif self.handicap_level == 3:
            # Level 3: Severe control reduction (70%)
            self.opponent_handicap_controls = 0.7
            log(f'[SELFPLAY] event=handicap level=3 controls=70%')

    def _check_selfplay_transition(self, stats=None):
        """Check if we should transition to dual self-play mode and save milestones.

        Uses avg_stage from C-level stats (actual worker data) with a confirmation
        gate: stage must be >= 19.9 for 3 consecutive checks before activating.
        """
        # Get current stage from stats (populated by C code via vec_log in workers)
        if stats and 'avg_stage' in stats and len(stats['avg_stage']) > 0:
            current_stage = min(stats['avg_stage'])
        elif stats and 'stage' in stats and len(stats['stage']) > 0:
            current_stage = min(stats['stage'])
        else:
            current_stage = 0  # No data yet — don't use driver_env (never stepped in MP)
            if self.trainer.epoch % 100 == 0:
                log(f'[SELFPLAY] warning=no_stage_in_stats epoch={self.trainer.epoch}')

        self._current_stage = current_stage

        # Check for milestone saves (stage 10, stage 20)
        self._check_milestone_save(int(current_stage))

        # Periodic logging so we can track curriculum progression
        if self.trainer.epoch % 50 == 0:
            log(f'[SELFPLAY] stage={current_stage:.2f} confirmed={self._selfplay_confirm_count}/3 selfplay={self.use_dual_selfplay}')

        if self.use_dual_selfplay:
            return  # Already in self-play mode

        # Trigger threshold: selfplay_min_stage=20, so threshold=19.9
        trigger_threshold = self.selfplay_min_stage - 0.1

        if current_stage >= trigger_threshold:
            self._selfplay_confirm_count += 1
            if self._selfplay_confirm_count < 3:
                log(f'[SELFPLAY] stage={current_stage:.2f} threshold={trigger_threshold} confirmed={self._selfplay_confirm_count}/3')
                return  # Not yet confirmed
        else:
            # Reset confirmation counter if stage drops
            if self._selfplay_confirm_count > 0:
                log(f'[SELFPLAY] stage_dropped={current_stage:.2f} confirm_reset_from={self._selfplay_confirm_count}')
            self._selfplay_confirm_count = 0
            return

        # 3 consecutive confirmations — activate self-play
        log(f'[SELFPLAY] event=activated stage={current_stage:.2f}')
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
        log(f'[SELFPLAY] event=recovery_hijacking_enabled')

        # Signal workers to enable opponent override via shared memory flag
        # (workers check this flag in step() before using opponent actions)
        if hasattr(self.vecenv, 'buf') and 'selfplay_active' in self.vecenv.buf:
            self.vecenv.buf['selfplay_active'][0] = 1
            log(f'[SELFPLAY] event=shm_flag_set selfplay_active=1')

        # Start vertical merge curriculum timer
        self._vertical_selfplay_start_step = self.trainer.global_step

        # Initialize ratchet rotation: start at rank 0 (stage10 = weakest)
        self._unlocked_rank = 0
        self._current_rotation_idx = 0
        self._epoch_start_step = self.trainer.global_step
        self._rank_mastery_streak = 0
        self._rotation_kills = 0.0
        self._rotation_episodes = 0.0
        self._epoch_kills = 0.0
        self._epoch_episodes = 0.0
        self._gate_clean_fights = 0.0
        self._gate_total_episodes = 0.0

        opponents = self._get_sorted_opponents()
        log(f'[RATCHET] event=init opponents={len(opponents)} tags={[t for _, t in opponents]}')
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
            # Process info and accumulate gate metrics
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.trainer.stats[k].extend(v)
                    else:
                        self.trainer.stats[k].append(v)
                # Accumulate gate metrics directly from each info dict
                if self.use_dual_selfplay and 'n' in i and 'perf' in i:
                    n_val = i['n']
                    if n_val > 0:
                        self._epoch_kills += i['perf'] * n_val
                        self._epoch_episodes += n_val
                        if self._current_opponent_tag != 'self':
                            sp_pk = i.get('sp_player_kills', 0)
                            sp_ok = i.get('sp_opp_kills', 0)
                            self._gate_player_kills += sp_pk * n_val
                            self._gate_opp_kills += sp_ok * n_val
                            clean_f = i.get('clean_fights', 0)
                            self._gate_clean_fights += clean_f * n_val
                            self._gate_total_episodes += n_val

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
        # Strength = ladder progress * current clean_fight_rate (drops when agent crashes)
        cf_vals = self.trainer.stats.get('clean_fights', [])
        current_cf_rate = np.mean(cf_vals) if cf_vals else 0.0
        strength = (float(self._unlocked_rank) / float(max(self.checkpoint_queue.max_checkpoints, 1))) * current_cf_rate
        self.trainer.stats['strength'] = [strength]

        if not self.use_dual_selfplay:
            # Standard single-perspective training
            # Check for selfplay transition BEFORE train() clears stats
            self._check_selfplay_transition(self.trainer.stats)
            logs = self.trainer.train()
            return logs

        # Dual self-play training
        logs = self._train_dual()

        # Progress vertical merge curriculum (level advancement + probability decay)
        self._update_vertical_curriculum()

        if self.league_opponent_pool is not None:
            # League mode: use only resample system for opponent selection.
            # Skip ratchet (epoch/rotation) to avoid two systems fighting over opponents.
            self._check_periodic_checkpoint()
            self._check_resample_opponent()
            self._update_opponent_win_rate(logs)
        else:
            # Standard self-play mode: ratchet + periodic checkpoints + stalemate
            self._check_epoch_boundary()
            self._check_periodic_checkpoint()
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
        # In league mode, skip opponent experience to avoid KL explosion:
        # opponent logprobs come from a different policy, causing massive importance
        # ratio divergence when the learner recomputes logprobs for those actions.
        if self.league_opponent_pool is not None and self._current_opponent_tag != 'self':
            # League mode with external opponent: train only on player experience
            combined_obs = self.trainer.observations
            combined_actions = self.trainer.actions
            combined_logprobs = self.trainer.logprobs
            combined_values = self.trainer.values
            combined_rewards = self.trainer.rewards
            combined_terminals = self.trainer.terminals
            combined_ratio = self.trainer.ratio
        else:
            # Standard self-play or playing against self: use both perspectives
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
        losses['gate_clean_fight_rate'] = self._gate_clean_fights / max(self._gate_total_episodes, 1)

        # 9. Strength: ladder progress * current clean_fight_rate (sweep metric)
        cf_vals = self.trainer.stats.get('clean_fights', [])
        current_cf_rate = np.mean(cf_vals) if cf_vals else 0.0
        strength = (float(self._unlocked_rank) / float(max(self.checkpoint_queue.max_checkpoints, 1))) * current_cf_rate
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
    log(f'[SELFPLAY] backend={backend}')

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

    init_log('league/logs', f'train_{run_id or "local"}')

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
    vertical_spawn_prob = float(args.get('env', {}).get('vertical_spawn_prob', DEFAULT_VERTICAL_PROB))

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
        vertical_prob=vertical_spawn_prob,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id
    )

    log(f'[TRAIN] event=start mode=dual_selfplay min_stage={selfplay_min_stage} perf_threshold={perf_threshold}')
    log(f'[TRAIN] ratchet epoch_length={opponent_epoch_length} mastery_streak={mastery_streak} pool_checkpoint_interval={pool_checkpoint_interval}')

    total_timesteps = train_config['total_timesteps']
    all_logs = []

    # Anchor evaluation config
    anchor_cfg = args.get('anchor_eval', {})
    anchor_eval_enabled = int(anchor_cfg.get('enabled', 0))
    anchor_eval_interval = int(anchor_cfg.get('interval', 50_000_000))
    anchor_games = int(anchor_cfg.get('games_per_anchor', 30))
    anchor_num_envs = int(anchor_cfg.get('num_envs', 16))
    anchor_dir = anchor_cfg.get('anchor_dir', 'pufferlib/ocean/dogfight/reference_opponents')
    last_anchor_eval_step = 0
    obs_scheme = args.get('env', {}).get('obs_scheme', 0)

    if anchor_eval_enabled:
        log(f'[TRAIN] anchor_eval enabled interval={anchor_eval_interval} games={anchor_games} dir={anchor_dir}')

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
            log(f'[TRAIN] mode={mode} step={trainer.global_step} queue={queue_len} opponent={opp_tag} rank={trainer._unlocked_rank} perf_ema={trainer._pool_perf_ema:.3f}')

        # Periodic anchor evaluation (runs during self-play phase only)
        if (anchor_eval_enabled
                and trainer.use_dual_selfplay
                and trainer.global_step - last_anchor_eval_step >= anchor_eval_interval):
            last_anchor_eval_step = trainer.global_step
            try:
                from pufferlib.ocean.dogfight.anchor_eval import evaluate_against_anchors
                import tempfile

                # Save current weights to temp file for evaluation
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                    torch.save(policy.state_dict(), f.name)
                    tmp_path = f.name

                device = train_config['device']
                anchor_results = evaluate_against_anchors(
                    model_path=tmp_path,
                    obs_scheme=obs_scheme,
                    anchor_dir=anchor_dir,
                    games_per_anchor=anchor_games,
                    num_envs=anchor_num_envs,
                    device=device,
                )
                os.unlink(tmp_path)

                # Inject anchor_rating into trainer stats for W&B logging
                anchor_rating = anchor_results.get('anchor_rating', 1000.0)
                trainer.trainer.stats['anchor_rating'] = [anchor_rating]

                # Also inject per-anchor win rates
                for tag, data in anchor_results.items():
                    if isinstance(data, dict) and 'win_rate' in data:
                        trainer.trainer.stats[f'anchor_wr_{tag}'] = [data['win_rate']]

                log(f'[ANCHOR] step={trainer.global_step} anchor_rating={anchor_rating:.0f} '
                    f'eval_time={anchor_results.get("eval_time", 0):.1f}s')

            except Exception as e:
                log(f'[ERROR] anchor_eval failed: {e}')

    # Cleanup
    model_path = trainer.close()
    if logger:
        logger.close(model_path)

    log(f'[TRAIN] event=complete')
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
                    log(f'[RATING] policy=candidate rating={result["elo"]:.0f} eval_time={result["eval_time_seconds"]:.1f}s')
                except Exception as e:
                    log(f'[ERROR] phase=elo_eval msg="{e}"')

        # Post-training anchor evaluation (gated by strength to save compute)
        # Below gate: linear estimate from strength (monotonic, instant)
        # Above gate: real eval against fixed anchors (~2 min)
        if all_logs:
            anchor_cfg = args.get('anchor_eval', {})
            strength_gate = float(anchor_cfg.get('strength_gate', 0.3))
            final_strength = all_logs[-1].get('environment/strength', 0.0)

            if model_path and final_strength >= strength_gate:
                try:
                    from pufferlib.ocean.dogfight.anchor_eval import evaluate_against_anchors
                    anchor_results = evaluate_against_anchors(
                        model_path=model_path,
                        obs_scheme=args['env'].get('obs_scheme', 0),
                        anchor_dir=anchor_cfg.get('anchor_dir', 'pufferlib/ocean/dogfight/reference_opponents'),
                        games_per_anchor=int(anchor_cfg.get('games_per_anchor', 30)),
                        num_envs=int(anchor_cfg.get('num_envs', 16)),
                        device=args['train']['device'],
                    )
                    anchor_rating = anchor_results.get('anchor_rating', 1000.0)
                    for entry in all_logs:
                        entry['environment/anchor_rating'] = anchor_rating
                    for atag, adata in anchor_results.items():
                        if isinstance(adata, dict) and 'win_rate' in adata:
                            all_logs[-1][f'environment/anchor_wr_{atag}'] = adata['win_rate']
                    log(f'[ANCHOR] event=post_training anchor_rating={anchor_rating:.0f} strength={final_strength:.3f}')
                except Exception as e:
                    log(f'[ERROR] phase=anchor_eval_post msg="{e}"')
                    # On error, fall back to linear estimate
                    anchor_rating = 200.0 + final_strength * 1200.0
                    for entry in all_logs:
                        entry['environment/anchor_rating'] = anchor_rating
            else:
                # Below gate or no model — linear estimate from strength
                anchor_rating = 200.0 + final_strength * 1200.0
                for entry in all_logs:
                    entry['environment/anchor_rating'] = anchor_rating
                log(f'[ANCHOR] event=estimated anchor_rating={anchor_rating:.0f} strength={final_strength:.3f} gate={strength_gate:.2f}')

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


def eval_selfplay(env_name, args, player_path, opponent_path, load_id=None, eval_spawn_mode=2):
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
    args['env']['eval_spawn_mode'] = eval_spawn_mode
    spawn_names = {0: 'random', 1: 'opp-advantage', 2: 'merge', 3: 'midfight'}
    print(f'[EVAL-SELFPLAY] Enabled curriculum_randomize, spawn mode={eval_spawn_mode} ({spawn_names.get(eval_spawn_mode, "unknown")})')

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

    # Win/loss/draw tracking
    player_wins = 0
    opponent_wins = 0
    draws = 0

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

            # Determine outcome from reward: +1 = player kill, -1 = player died, else timeout
            ep_reward = float(reward.flatten()[0]) if hasattr(reward, 'flatten') else float(reward)
            if ep_reward > 0.5:
                player_wins += 1
                outcome = 'PLAYER WINS'
            elif ep_reward < -0.5:
                opponent_wins += 1
                outcome = 'OPPONENT WINS'
            else:
                draws += 1
                outcome = 'DRAW/TIMEOUT'

            total = player_wins + opponent_wins + draws
            p_pct = 100 * player_wins / total
            o_pct = 100 * opponent_wins / total
            d_pct = 100 * draws / total
            print(f'[EVAL] Ep {episode_count}: {outcome} | P:{player_wins} O:{opponent_wins} D:{draws} ({p_pct:.0f}%/{o_pct:.0f}%/{d_pct:.0f}%)')

            # Don't call vecenv.reset() here — c_step() already calls c_reset()
            # internally. A second reset would overwrite last_winner/last_death_reason,
            # which the HUD uses to display "PLAYER WINS" / "OPPONENT WINS".

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


def train_league_round(policy_entry, manifest, league_dir, training_steps,
                       device='cuda', wandb_project=None):
    """Train one policy for one league round.

    Loads the policy, creates env with the correct obs_scheme, builds
    league/antiforgetting pools from manifest, and trains with
    skip_curriculum=True and 35/50/15 split.

    Args:
        policy_entry: PolicyEntry from manifest.
        manifest: LeagueManifest (for building opponent pools).
        league_dir: Base directory for resolving model paths.
        training_steps: Number of training steps.
        device: Torch device string.
        wandb_project: Optional W&B project for logging.

    Returns:
        Path to candidate checkpoint, or None on failure.
    """
    env_name = 'puffer_dogfight'
    # load_config calls argparse on sys.argv — save/restore to avoid
    # conflicts with league.py's CLI args
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = orig_argv

    init_log('league/logs', f'league_train_{policy_entry.id}')

    # Override env config for league training
    args['env']['obs_scheme'] = policy_entry.obs_scheme
    args['env']['fixed_stage'] = 20
    args['env']['curriculum_enabled'] = 1
    args['train']['total_timesteps'] = training_steps
    args['train']['device'] = device

    # Apply policy-specific training config if available
    for k, v in policy_entry.config.items():
        if k in args.get('train', {}):
            args['train'][k] = v
        elif k in args.get('env', {}):
            args['env'][k] = v

    # Build league opponent pool (same obs_scheme, excluding self)
    league_pool = []
    antiforgetting = []
    same_scheme = manifest.get_policies_by_scheme(policy_entry.obs_scheme)
    for p in same_scheme:
        if p.id == policy_entry.id:
            continue
        model_path = os.path.join(league_dir, p.model_path)
        if not os.path.exists(model_path):
            log(f'[ERROR] policy={p.id} phase=league_train msg="Missing model {model_path}"')
            continue
        if p.status == 'frozen':
            antiforgetting.append((model_path, p.id))
        else:
            league_pool.append((model_path, p.id))

    # Include frozen anchors in league pool too (for PFSP diversity)
    for path, tag in antiforgetting:
        league_pool.append((path, tag))

    log(f'[TRAIN] policy={policy_entry.id} obs_scheme={policy_entry.obs_scheme} hidden_size={policy_entry.hidden_size}')
    log(f'[TRAIN] league_pool={len(league_pool)} antiforgetting={len(antiforgetting)}')

    # Create environment
    vecenv = pufferl.load_env(env_name, args)

    # Create policy with correct hidden_size
    policy = pufferl.load_policy(args, vecenv, env_name)

    # Load existing weights
    model_path = os.path.join(league_dir, policy_entry.model_path)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and 'policy_state_dict' in state_dict:
        state_dict = state_dict['policy_state_dict']
    # Try direct load first (PuffeRL format with policy./lstm. prefixes),
    # fall back to stripping prefixes for raw Default policy format
    try:
        policy.load_state_dict(state_dict)
    except RuntimeError:
        cleaned = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '').replace('policy.', '')
            if new_k.startswith('lstm.') or new_k.startswith('cell.'):
                continue
            cleaned[new_k] = v
        policy.load_state_dict(cleaned)
    log(f'[TRAIN] event=loaded_weights path={model_path}')

    # Create logger
    logger = None
    run_id = None
    if wandb_project:
        args['wandb'] = True
        args['wandb_project'] = wandb_project
        logger = pufferl.WandbLogger(args)
        if hasattr(logger, 'run') and logger.run:
            run_id = logger.run.id

    # Read league config (opponent split + resample interval)
    league_args = args.get('league', {})
    sp_prob = float(league_args.get('self_play_prob', 0.60))
    lg_prob = float(league_args.get('league_prob', 0.30))
    af_prob = float(league_args.get('antiforgetting_prob', 0.10))
    resample_interval = int(league_args.get('opponent_resample_interval', 3_000_000))

    training_steps = args['train'].get('total_timesteps', '?')
    log(f'[TRAIN] opponent_split self_play={sp_prob:.2f} league={lg_prob:.2f} antiforgetting={af_prob:.2f}')

    # Override LR for league fine-tuning. Main training uses CosineAnnealingLR
    # over 200-600M steps, so the policy ended at near-zero LR. Restarting at
    # full LR destabilizes converged weights.
    league_lr = league_args.get('league_lr')
    if league_lr is not None:
        args['train']['learning_rate'] = float(league_lr)
        args['train']['anneal_lr'] = False  # Constant LR for fine-tuning

    actual_lr = args['train'].get('learning_rate', '?')
    anneal = args['train'].get('anneal_lr', True)
    log(f'[TRAIN] config lr={actual_lr} anneal_lr={anneal} total_steps={training_steps} resample_interval={resample_interval}')

    # Create trainer with skip_curriculum and league pools
    train_config = {**args['train'], 'env': env_name}
    trainer = DualPerspectiveTrainer(
        train_config, vecenv, policy, logger,
        skip_curriculum=True,
        league_opponent_pool=league_pool if league_pool else None,
        antiforgetting_pool=antiforgetting if antiforgetting else None,
        self_play_prob=sp_prob,
        league_prob=lg_prob,
        antiforgetting_prob=af_prob,
        opponent_resample_interval=resample_interval,
        checkpoint_dir=f'checkpoints/league_{policy_entry.id}',
        run_id=run_id,
    )

    # Training loop
    total_timesteps = train_config['total_timesteps']
    last_log_step = 0
    LOG_INTERVAL = 5_000_000

    # Collapse detection: abort if policy sustains low perf for too long
    COLLAPSE_PERF_THRESHOLD = 0.15
    COLLAPSE_STREAK_LIMIT = 3  # consecutive bad checks → abort
    COLLAPSE_CHECK_INTERVAL = 10_000_000
    low_perf_streak = 0
    last_collapse_check_step = 0
    early_stopped = False

    # Best checkpoint tracking: save state with highest pool_perf
    BEST_CHECKPOINT_WARMUP = 10_000_000
    best_pool_perf = -1.0
    best_state_dict = None
    best_step = 0
    BEST_CHECK_INTERVAL = 5_000_000
    last_best_check_step = 0

    last_logs = None
    while trainer.global_step < total_timesteps:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        trainer.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = trainer.train()
        if logs:
            last_logs = logs

        # Periodic progress log
        if trainer.global_step - last_log_step >= LOG_INTERVAL:
            last_log_step = trainer.global_step
            stats = {
                'step': trainer.global_step,
                'opponent': trainer._current_opponent_tag or '?',
            }
            if trainer.opponent_win_rates:
                wr_strs = ' '.join(f'{t}={w:.2f}' for t, w in sorted(trainer.opponent_win_rates.items()))
                stats['win_rates'] = wr_strs
            if last_logs:
                for key in ('losses/policy_loss', 'losses/value_loss', 'losses/entropy', 'losses/old_approx_kl'):
                    if key in last_logs:
                        short = key.split('/')[-1]
                        stats[short] = f'{last_logs[key]:.4f}'
                if 'environment/perf' in last_logs:
                    stats['perf'] = f'{last_logs["environment/perf"]:.3f}'
            log(f'[TRAIN] event=progress ' + ' '.join(f'{k}={v}' for k, v in stats.items()))

        # Best checkpoint tracking
        if (trainer.global_step >= BEST_CHECKPOINT_WARMUP
                and trainer.global_step - last_best_check_step >= BEST_CHECK_INTERVAL):
            last_best_check_step = trainer.global_step
            current_pool_perf = trainer.pool_perf
            if current_pool_perf > best_pool_perf:
                best_pool_perf = current_pool_perf
                best_state_dict = copy.deepcopy(policy.state_dict())
                best_step = trainer.global_step
                log(f'[TRAIN] event=new_best pool_perf={best_pool_perf:.3f} step={best_step}')

        # Collapse detection: NaN or sustained low perf
        if last_logs:
            value_loss = last_logs.get('losses/value_loss')
            if value_loss is not None and (value_loss != value_loss):  # NaN check
                log(f'[TRAIN] event=collapse reason=nan_value_loss step={trainer.global_step}')
                early_stopped = True
                break

            if trainer.global_step - last_collapse_check_step >= COLLAPSE_CHECK_INTERVAL:
                last_collapse_check_step = trainer.global_step
                perf = last_logs.get('environment/perf', 1.0)
                if perf < COLLAPSE_PERF_THRESHOLD:
                    low_perf_streak += 1
                    log(f'[TRAIN] event=low_perf streak={low_perf_streak}/{COLLAPSE_STREAK_LIMIT} perf={perf:.3f} step={trainer.global_step}')
                    if low_perf_streak >= COLLAPSE_STREAK_LIMIT:
                        log(f'[TRAIN] event=collapse reason=sustained_low_perf streak={low_perf_streak} perf={perf:.3f} step={trainer.global_step}')
                        early_stopped = True
                        break
                else:
                    if low_perf_streak > 0:
                        log(f'[TRAIN] event=perf_recovered streak_was={low_perf_streak} perf={perf:.3f} step={trainer.global_step}')
                    low_perf_streak = 0

    if early_stopped:
        log(f'[TRAIN] event=early_stop policy={policy_entry.id} step={trainer.global_step}')

    # Training summary
    if trainer.opponent_win_rates:
        log(f'[TRAIN] event=summary policy={policy_entry.id} steps={trainer.global_step}')
        for tag, wr in sorted(trainer.opponent_win_rates.items(), key=lambda x: x[1]):
            log(f'[TRAIN] event=final_wr opponent={tag} wr={wr:.3f}')
    log(f'[TRAIN] event=best_summary best_step={best_step} best_pool_perf={best_pool_perf:.3f} final_pool_perf={trainer.pool_perf:.3f}')

    # Save candidate checkpoint (best weights if available, otherwise final)
    candidate_dir = os.path.join(league_dir, 'candidates')
    os.makedirs(candidate_dir, exist_ok=True)
    next_gen = policy_entry.generation + 1
    candidate_filename = f'{policy_entry.id}_gen{next_gen}_candidate.pt'
    candidate_path = os.path.join(candidate_dir, candidate_filename)
    if best_state_dict is not None:
        torch.save(best_state_dict, candidate_path)
        log(f'[CHECKPOINT] event=candidate policy={policy_entry.id} source=best best_step={best_step} best_pool_perf={best_pool_perf:.3f} path={candidate_path}')
    else:
        torch.save(policy.state_dict(), candidate_path)
        log(f'[CHECKPOINT] event=candidate policy={policy_entry.id} source=final step={trainer.global_step} path={candidate_path}')

    # Cleanup
    model_path_result = trainer.close()
    if logger:
        logger.close(model_path_result)
    return candidate_path


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
        eval_spawn_mode = 2  # Default to merge (symmetric, fair)

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

            # --eval-spawn-mode <0|1|2|3>
            if arg == '--eval-spawn-mode':
                if i + 1 < len(sys.argv):
                    eval_spawn_mode = int(sys.argv[i + 1])
                    i += 2
                    continue
            elif arg.startswith('--eval-spawn-mode='):
                eval_spawn_mode = int(arg.split('=', 1)[1])
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

        spawn_names = {0: 'random', 1: 'opp-advantage', 2: 'merge', 3: 'midfight'}
        print(f'[EVAL-SELFPLAY] Starting eval mode')
        print(f'[EVAL-SELFPLAY] Player model: {player_path or load_id}')
        print(f'[EVAL-SELFPLAY] Opponent checkpoint: {opponent_checkpoint}')
        print(f'[EVAL-SELFPLAY] Spawn mode: {eval_spawn_mode} ({spawn_names.get(eval_spawn_mode, "unknown")})')

        # Run eval
        eval_selfplay(env_name, args, player_path, opponent_checkpoint, load_id, eval_spawn_mode)
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
