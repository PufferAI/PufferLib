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
DEFAULT_PERF_THRESHOLD = 0.65  # Kill rate to trigger checkpoint save + opponent upgrade
DEFAULT_MIN_STEPS_BETWEEN_CHECKPOINTS = 2_000_000  # Minimum steps before saving new checkpoint
DEFAULT_MAX_CHECKPOINTS = 20  # Max selfplay checkpoints (milestones always kept)


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
                 checkpoint_dir=None,
                 run_id=None):
        # Store custom config
        self.opponent_update_interval = opponent_update_interval
        self.selfplay_min_stage = selfplay_min_stage
        self.checkpoint_lag = checkpoint_lag
        self.perf_threshold = perf_threshold
        self.min_steps_between_checkpoints = min_steps_between_checkpoints
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
        self.last_checkpoint_step = 0
        self._current_stage = 0

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
        """Load opponent from checkpoint queue instead of copying weights."""
        opponent_path = self.checkpoint_queue.get_opponent(lag=self.checkpoint_lag)

        if opponent_path is None:
            # Queue too small, fall back to copying learner weights
            self.opponent_policy.load_state_dict(self.learner_policy.state_dict())
            self.last_opponent_update = self.trainer.global_step
            print(f'[DUAL-SELFPLAY] Queue too small, copied learner weights at step {self.trainer.global_step}')
            return

        if opponent_path != self._current_opponent_path:
            self._load_opponent_from_checkpoint(opponent_path)
            self._current_opponent_path = opponent_path
            self.last_opponent_update = self.trainer.global_step

    def _load_opponent_from_checkpoint(self, checkpoint_path: str):
        """Load opponent policy from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
        self.opponent_policy.load_state_dict(checkpoint['policy_state_dict'])

        # Get checkpoint info for logging
        tag = checkpoint.get('tag', 'unknown')
        step = checkpoint.get('step', 0)
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
            print(f'[CHECKPOINT-QUEUE] Saved milestone: stage20 at step {self.trainer.global_step}')

    def _check_domination(self, logs):
        """Check if learner dominates opponent using perf metric.

        When the learner's kill rate (perf) exceeds the threshold:
        1. Save a new checkpoint
        2. Upgrade opponent to older checkpoint (lag positions behind)
        """
        if not self.use_dual_selfplay:
            return

        # Check minimum steps since last checkpoint
        steps_since_last = self.trainer.global_step - self.last_checkpoint_step
        if steps_since_last < self.min_steps_between_checkpoints:
            return

        # Get perf from logs (already computed kill rate)
        perf = logs.get('perf', 0) if logs else 0
        if perf >= self.perf_threshold:
            print(f'[CHECKPOINT-QUEUE] Learner dominating (perf={perf:.2f} >= {self.perf_threshold}), saving checkpoint')

            # Save new checkpoint
            checkpoint_num = len([c for c in self.checkpoint_queue.checkpoints if not c.is_milestone()])
            tag = f"selfplay_{checkpoint_num}"
            self.checkpoint_queue.save(
                self.learner_policy,
                self.trainer.global_step,
                self._current_stage,
                tag
            )
            self.last_checkpoint_step = self.trainer.global_step

            # Log queue state
            queue_state = self.checkpoint_queue.get_queue_state()
            print(f'[CHECKPOINT-QUEUE] Queue: {queue_state["tags"]}')

            # Upgrade opponent to older checkpoint (lag positions behind)
            self._update_opponent()

    def _check_selfplay_transition(self):
        """Check if we should transition to dual self-play mode and save milestones."""
        # Get current stage from driver env (update tracking variable)
        current_stage = getattr(self.driver_env, '_current_stage', 0)
        self._current_stage = current_stage

        # Check for milestone saves (stage 10, stage 20)
        self._check_milestone_save(int(current_stage))

        debug(1, f'_check_selfplay_transition: use_dual_selfplay={self.use_dual_selfplay}')
        if self.use_dual_selfplay:
            return  # Already in self-play mode

        debug(1, f'_check_selfplay_transition: stage={current_stage}, min={self.selfplay_min_stage}')

        if current_stage >= self.selfplay_min_stage:
            print(f'[DUAL-SELFPLAY] Transitioning to dual self-play at stage {current_stage}', flush=True)
            self.use_dual_selfplay = True

            # Allocate opponent buffers
            self._allocate_opponent_buffers()

            # Enable opponent override in C code
            # With Multiprocessing: override is already enabled in dogfight.py when opponent buffers are set
            # With Serial: enable it now via direct C binding call
            if not hasattr(self.vecenv, 'buf') or 'opponent_actions' not in self.vecenv.buf:
                from pufferlib.ocean.dogfight import binding
                binding.vec_enable_opponent_override(self.driver_env.c_envs, 1)

            # Initialize opponent from checkpoint queue
            self._update_opponent()

    def evaluate(self):
        """Evaluate with dual experience collection in self-play mode."""
        self._check_selfplay_transition()

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
        if not self.use_dual_selfplay:
            # Standard single-perspective training
            logs = self.trainer.train()
            # Still check milestones during curriculum phase
            self._check_selfplay_transition()
            return logs

        # Dual self-play training
        logs = self._train_dual()

        # Check if learner dominates opponent -> save checkpoint and upgrade
        self._check_domination(logs)

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
        losses['dual_selfplay'] = 1.0  # Flag for logging

        profile.end()
        logs = None
        self.trainer.epoch += 1
        done_training = self.trainer.global_step >= config['total_timesteps']
        if done_training or self.trainer.global_step == 0 or time.time() > self.trainer.last_log_time + 0.25:
            logs = self.trainer.mean_and_log()
            self.trainer.losses = losses
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
    # Set load_model_path/load_id so load_policy picks it up
    if player_path:
        args['load_model_path'] = player_path
    if load_id:
        args['load_id'] = load_id

    player_policy = pufferl.load_policy(args, vecenv, env_name)
    player_policy.eval()

    # Create opponent policy (same architecture, different weights)
    device = args['train']['device']
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
            action_o_np = action_o.cpu().numpy().reshape(vecenv.action_space.shape)

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

        # Set opponent actions in C (before stepping) - slice to match actual num_agents
        binding.vec_set_opponent_actions(driver.c_envs, action_o_np[:num_agents])

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

    # Load standard dogfight config
    args = pufferl.load_config(env_name)

    # NOTE: Dual self-play now works with Multiprocessing backend!
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
        checkpoint_dir=checkpoint_dir,
        run_id=run_id
    )

    print(f'[DUAL-SELFPLAY] Starting training with checkpoint queue')
    print(f'[DUAL-SELFPLAY] Min stage for self-play: {selfplay_min_stage}')
    print(f'[DUAL-SELFPLAY] Checkpoint lag: {checkpoint_lag} (opponent is {checkpoint_lag} checkpoint(s) behind)')
    print(f'[DUAL-SELFPLAY] Perf threshold: {perf_threshold} (save checkpoint when perf >= this)')
    print(f'[DUAL-SELFPLAY] Min steps between checkpoints: {min_steps_between_checkpoints}')

    # Training loop
    while trainer.global_step < train_config['total_timesteps']:
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        trainer.evaluate()
        if train_config['device'] == 'cuda':
            torch.compiler.cudagraph_mark_step_begin()
        logs = trainer.train()

        # Log dual self-play status periodically
        if trainer.epoch % 100 == 0 and trainer.epoch > 0:
            mode = "DUAL" if trainer.use_dual_selfplay else "CURRICULUM"
            queue_len = len(trainer.checkpoint_queue)
            opponent_entry = trainer.checkpoint_queue.get_opponent_entry(trainer.checkpoint_lag)
            opponent_tag = opponent_entry.tag if opponent_entry else "none"
            print(f'[DUAL-SELFPLAY] Mode: {mode}, Steps: {trainer.global_step}, '
                  f'Queue: {queue_len} checkpoints, Opponent: {opponent_tag}')

    # Cleanup
    model_path = trainer.close()
    if logger:
        logger.close(model_path)

    print(f'[DUAL-SELFPLAY] Training complete')


if __name__ == '__main__':
    main()
