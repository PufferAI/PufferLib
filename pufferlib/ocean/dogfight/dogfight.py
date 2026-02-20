import time
import numpy as np
import gymnasium
import torch

import pufferlib
from pufferlib.ocean.dogfight import binding
from pufferlib.ocean.dogfight.dogfight_log import log as dogfight_log


# Autopilot mode constants (must match autopilot.h enum)
class AutopilotMode:
    STRAIGHT = 0
    LEVEL = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    CLIMB = 4
    DESCEND = 5
    HARD_TURN_LEFT = 6
    HARD_TURN_RIGHT = 7
    WEAVE = 8
    EVASIVE = 9
    RANDOM = 10


# Observation sizes by scheme (must match C OBS_SIZES in dogfight.h)
# All schemes include timer observation (tick/max_steps) at the end
OBS_SIZES = {
    0: 17,   # MOMENTUM_GFORCE: G-force awareness (proven winner from df24)
    1: 22,   # PILOT: Pilot awareness + energy
    2: 22,   # RATES_LEAN: Scheme 0 + tactical rates
    3: 27,   # RATES_FULL: Scheme 1 + tactical rates
}


class Dogfight(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        render_mode=None,
        render_fps=None,
        report_interval=1,
        buf=None,
        seed=42,
        max_steps=3000,
        obs_scheme=0,

        curriculum_enabled=0,
        curriculum_randomize=0,
        eval_spawn_mode=0,  # 0=random, 1=opponent_advantage (opponent behind player)
        fixed_stage=-1,
        max_stage=20,       # Allow curriculum to reach AutoAce/self-play (stage 20)
        eval_interval=2_500_000,    # Steps between curriculum evaluations (2.5M = ~1s at 2.5M SPS)
        warmup_steps=3_000_000,     # Steps before curriculum starts evaluating (3M = ~1.2s at 2.5M SPS)
        min_eval_episodes=50,       # Minimum episodes in window before evaluating mastery
        # Finalization: snap to mastered stage at end of training
        total_timesteps=200_000_000,  # Total training steps (for finalization calculation)
        num_workers=8,                # Number of parallel workers (for finalization calculation)
        finalize_margin=50_000_000,   # Start finalization this many steps before end
        # df11: Simplified rewards (6 terms)
        reward_aim_scale=0.05,       # Continuous aiming reward
        reward_closing_scale=0.003,  # Per m/s closing
        penalty_neg_g=0.02,          # Enforce "pull to turn"
        speed_min=50.0,              # Stall threshold
        control_rate_penalty=0.0,    # Penalty for action rate changes (sweep to find optimal)
        low_altitude_threshold=1500.0,  # Altitude below which penalty applies
        low_altitude_penalty=0.01,      # Penalty scale at ground level
        aim_decay_stage=15.0,        # Stage at which aim reward reaches 0 (anti-spiral) - DEPRECATED
        # Timestep-based shaping decay: anneals r_aim and r_closing during self-play
        shaping_decay_start=100_000_000,  # Start annealing at this global step
        shaping_decay_end=150_000_000,    # Complete annealing at this global step
        # Self-play: load frozen checkpoint as opponent
        opponent_checkpoint=None,    # Path to .pt checkpoint file
        opponent_device='cpu',       # Device for opponent policy inference
        # Self-play: policy pool for skill-based opponent selection
        policy_pool=None,            # PolicyPool instance (optional)
        opponent_selection='skill_match',  # Selection strategy: skill_match, prioritized, random, latest
        opponent_swap_interval=500_000,    # Steps between opponent swaps
        # Recovery hijacking (breaks death spiral equilibrium in self-play)
        recovery_enabled=1,
        recovery_altitude_threshold=500.0,
        recovery_trigger_prob=0.1,
        recovery_speed_threshold=70.0,
        recovery_bank_deg=60.0,
        # Domain randomization: 0.0 = off, 0.1 = +/-10% physics variation per-episode
        domain_randomization=0.0,
        # Consumed by train_dual_selfplay.py, accepted here to avoid unknown-kwarg error
        vertical_spawn_prob=0.10,
    ):
        # Observation size depends on scheme
        obs_size = OBS_SIZES.get(obs_scheme, 19)
        self.obs_scheme = obs_scheme
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Action: Box(5) continuous [-1, 1]
        # [0] throttle, [1] elevator, [2] ailerons, [3] rudder, [4] trigger
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32
        )

        self.num_agents = num_envs
        self.agents_per_batch = num_envs  # For pufferl LSTM compatibility
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.report_interval = report_interval
        self.tick = 0

        # Global curriculum state (step-based window evaluation)
        self._current_stage = 0
        self._target_stage = 0.9           # Start at 0.9 (90% stage 1, 10% stage 0)
        self._warmup_steps = warmup_steps  # Steps before curriculum starts evaluating
        self._eval_interval = eval_interval  # Steps between curriculum evaluations
        self._last_eval_step = warmup_steps  # First eval at warmup + eval_interval
        self.curriculum_enabled = curriculum_enabled
        self.fixed_stage = fixed_stage
        self.max_stage = max_stage
        print(f'[DOGFIGHT] Curriculum max_stage={self.max_stage}')
        self.min_eval_episodes = min_eval_episodes

        # Mastered stage tracking (pure mastery-gated progression)
        self._mastered_stage = -1  # Highest stage with perf >= 0.90 AND >= 250 episodes (-1 = none)

        # Finalization: snap to mastered stage near end of training
        # total_timesteps is global total, finalize_margin is global margin
        # Per-worker finalize step = (global_total - global_margin) / num_workers
        self._finalize_margin = finalize_margin
        self._finalize_at_steps = (total_timesteps - finalize_margin) // num_workers
        #print(f'[CURRICULUM] Initialized: finalize_at={self._finalize_at_steps}, mastered_stage={self._mastered_stage}')

        # Base stage tracking: performance at int(curriculum_target) only
        self._base_stage_kills = 0.0
        self._base_stage_eps = 0.0

        # If fixed_stage is set, lock to that stage
        if fixed_stage >= 0:
            self._target_stage = float(fixed_stage)
            self._current_stage = fixed_stage

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)  # REQUIRED for continuous

        self._env_handles = []
        for env_num in range(num_envs):
            handle = binding.env_init(
                self.observations[env_num:(env_num+1)],
                self.actions[env_num:(env_num+1)],
                self.rewards[env_num:(env_num+1)],
                self.terminals[env_num:(env_num+1)],
                self.truncations[env_num:(env_num+1)],
                env_num,
                env_num=env_num,
                report_interval=self.report_interval,
                max_steps=max_steps,
                obs_scheme=obs_scheme,

                curriculum_enabled=curriculum_enabled,
                curriculum_randomize=curriculum_randomize,
                eval_spawn_mode=eval_spawn_mode,

                reward_aim_scale=reward_aim_scale,
                reward_closing_scale=reward_closing_scale,
                penalty_neg_g=penalty_neg_g,
                speed_min=speed_min,
                control_rate_penalty=control_rate_penalty,
                low_altitude_threshold=low_altitude_threshold,
                low_altitude_penalty=low_altitude_penalty,
                aim_decay_stage=aim_decay_stage,
                shaping_decay_start=shaping_decay_start,
                shaping_decay_end=shaping_decay_end,
                # Domain randomization
                domain_randomization=domain_randomization,
                # Recovery hijacking config
                recovery_enabled=recovery_enabled,
                recovery_altitude_threshold=recovery_altitude_threshold,
                recovery_trigger_prob=recovery_trigger_prob,
                recovery_speed_threshold=recovery_speed_threshold,
                recovery_bank_deg=recovery_bank_deg,
            )
            self._env_handles.append(handle)

        self.c_envs = binding.vectorize(*self._env_handles)

        # Set opponent observation/reward/action buffers if provided (for dual self-play with Multiprocessing)
        # These buffers come from shared memory in Multiprocessing backend
        self._opponent_observations = None
        self._opponent_rewards = None
        self._opponent_actions = None
        if buf is not None and 'opponent_observations' in buf:
            self._opponent_observations = buf['opponent_observations']
            self._opponent_rewards = buf['opponent_rewards']
            # Flatten to match C expectations: shape (num_envs * obs_size,) and (num_envs,)
            opp_obs_flat = self._opponent_observations.reshape(-1)
            opp_rew_flat = self._opponent_rewards.reshape(-1)
            binding.vec_set_opponent_buffers(self.c_envs, opp_obs_flat, opp_rew_flat)
            # NOTE: Don't enable opponent override here - let DualPerspectiveTrainer
            # control when to activate self-play mode. Otherwise sp_* stats get
            # logged during curriculum training which is confusing.
        if buf is not None and 'opponent_actions' in buf:
            self._opponent_actions = buf['opponent_actions']
        # Shared flag indicating self-play mode is active (set by main process)
        self._selfplay_active = None
        self._opponent_override_enabled = False  # Track if we've enabled C-side override
        if buf is not None and 'selfplay_active' in buf:
            self._selfplay_active = buf['selfplay_active']

        # Self-play: opponent policy (loaded after c_envs created)
        self.opponent_policy = None
        self.opponent_device = opponent_device
        self.opponent_lstm_state = None  # For recurrent policies (future)
        self._current_opponent_path = None  # Track current opponent for swap detection

        # Policy pool: skill-based opponent selection
        self.policy_pool = policy_pool
        self.opponent_selection = opponent_selection
        self.opponent_swap_interval = opponent_swap_interval
        self._last_opponent_swap = 0  # Steps since last swap
        self._save_to_pool_callback = None  # Set by training script to save checkpoints

        if opponent_checkpoint:
            self._load_opponent_policy(opponent_checkpoint)
            self._current_opponent_path = opponent_checkpoint

        # Set fixed stage on C side if specified
        if fixed_stage >= 0:
            binding.vec_set_curriculum_target(self.c_envs, float(fixed_stage))

    def _load_opponent_policy(self, path):
        """Load a frozen checkpoint as the opponent policy."""
        # Create policy with same architecture as training
        from pufferlib.ocean.torch import DogfightPolicy
        self.opponent_policy = DogfightPolicy(self)
        self.opponent_policy = self.opponent_policy.to(self.opponent_device)

        # Load checkpoint weights
        state_dict = torch.load(path, map_location=self.opponent_device, weights_only=True)

        # Handle different checkpoint formats:
        # 1. Strip 'module.' prefix from distributed training
        # 2. Strip 'policy.' prefix from LSTMWrapper
        # 3. Skip LSTM-specific keys (lstm.*, cell.*)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Skip LSTM keys - we only load the base policy
            if k.startswith('lstm.') or k.startswith('cell.'):
                continue
            # Strip prefixes
            new_k = k.replace('module.', '').replace('policy.', '')
            cleaned_state_dict[new_k] = v

        self.opponent_policy.load_state_dict(cleaned_state_dict)

        # Freeze for inference only
        self.opponent_policy.eval()
        for p in self.opponent_policy.parameters():
            p.requires_grad = False

        # Enable C-side override mode (use external actions instead of autopilot)
        binding.vec_enable_opponent_override(self.c_envs, 1)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed else 0)
        # Reset opponent LSTM state on episode reset (for recurrent policies)
        self.opponent_lstm_state = None
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        # Check if main process has signaled self-play mode via shared memory flag
        # This enables opponent override AND recovery hijacking in workers (Multiprocessing)
        if self._selfplay_active is not None and self._selfplay_active[0] == 1:
            if not self._opponent_override_enabled:
                binding.vec_enable_opponent_override(self.c_envs, 1)
                binding.vec_set_selfplay_active(self.c_envs, 1)  # Enable recovery hijacking
                self._opponent_override_enabled = True

        # Self-play: read opponent actions from shared memory buffer (dual self-play with Multiprocessing)
        # or compute from local frozen policy (standard self-play with Serial)
        if self._opponent_actions is not None and self._opponent_override_enabled:
            # Multiprocessing dual self-play: read opponent actions from shared memory
            # Main process writes actions to buf, workers read them here
            opp_actions_flat = self._opponent_actions.reshape(-1, 5)  # Shape: (num_envs, 5)
            binding.vec_set_opponent_actions(self.c_envs, opp_actions_flat)
        elif self.opponent_policy is not None:
            # Serial self-play: compute opponent actions from frozen policy in-process
            opp_obs = binding.vec_get_opponent_observations(self.c_envs)
            opp_obs_t = torch.as_tensor(opp_obs, device=self.opponent_device)

            with torch.no_grad():
                # Policy returns Normal distribution for continuous actions
                logits, _ = self.opponent_policy.forward_eval(
                    opp_obs_t, state=self.opponent_lstm_state
                )
                # Sample actions from the distribution
                opp_actions = logits.sample()
                opp_actions = opp_actions.cpu().numpy().astype(np.float32)

            binding.vec_set_opponent_actions(self.c_envs, opp_actions)

        self.tick += 1
        # Update global step for shaping reward decay (tick * num_agents is approximate global step)
        binding.vec_set_global_step(self.c_envs, self.tick * self.num_agents)
        binding.vec_step(self.c_envs)

        # Auto-render if render_mode is 'human' (Gymnasium convention)
        if self.render_mode == 'human':
            self.render()
            if self.render_fps:
                time.sleep(1.0 / self.render_fps)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

                # Curriculum v4: Pure Mastery-Gated Progression
                # Target is ALWAYS mastered_stage + 0.9 (except during finalization)
                # No advancement logic - target changes ONLY when mastery is achieved
                # Skip progression if fixed_stage is set (testing mode)
                if self.curriculum_enabled and self.fixed_stage < 0:
                    n = log_data.get('n', 0)        # episodes completed this tick
                    total_steps = self.tick * self.num_agents

                    # Only accumulate AFTER warmup (avoid early kill bias)
                    if total_steps >= self._warmup_steps:
                        # Track base stage performance (for mastery gating)
                        if n > 0:
                            base_kills = log_data.get('base_stage_kills', 0)
                            base_eps = log_data.get('base_stage_eps', 0)
                            self._base_stage_kills += base_kills
                            self._base_stage_eps += base_eps

                        # Evaluate at intervals
                        if total_steps - self._last_eval_step >= self._eval_interval:
                            self._last_eval_step = total_steps

                            # Compute base stage performance (current stage mastery)
                            base_stage_perf = (self._base_stage_kills / self._base_stage_eps) if self._base_stage_eps > 0 else 0.0

                            # Check mastery at MAJORITY stage (round, not floor)
                            # At target 0.9, majority is stage 1 (90% of episodes)
                            mastery_stage = round(self._target_stage)
                            if base_stage_perf >= 0.90 and self._base_stage_eps >= self.min_eval_episodes:
                                if mastery_stage > self._mastered_stage:
                                    dogfight_log(f'[CURRICULUM] event=mastered stage={mastery_stage} perf={base_stage_perf:.3f} eps={self._base_stage_eps:.0f}')
                                    self._mastered_stage = mastery_stage
                                    # Reset base stage tracking for new level
                                    self._base_stage_kills = 0.0
                                    self._base_stage_eps = 0.0

                                    # Save milestone checkpoint to pool if callback is set
                                    if self._save_to_pool_callback is not None:
                                        self._save_to_pool_callback(mastery_stage, base_stage_perf)

                            # Target is ALWAYS mastered + 0.9 (except finalization)
                            # Cap at max_stage to avoid AutoAce/self-play if desired
                            in_finalization = self._finalize_margin > 0 and total_steps >= self._finalize_at_steps
                            if in_finalization:
                                # Lock at stage 19+ once reached (self-play is 50/50, can't master)
                                if self._mastered_stage >= 19:
                                    new_target = 20.0  # Lock at self-play forever
                                else:
                                    new_target = float(self._mastered_stage) + 0.01
                            else:
                                new_target = float(self._mastered_stage) + 0.9

                            # Never drop below 19.0 once stage 19+ is reached
                            if self._mastered_stage >= 19:
                                new_target = max(new_target, 19.0)

                            new_target = min(new_target, float(self.max_stage))

                            if abs(self._target_stage - new_target) > 0.01:
                                dogfight_log(f'[CURRICULUM] event=target old={self._target_stage:.2f} new={new_target:.2f} mastered={self._mastered_stage}')
                                self._target_stage = new_target
                                self._current_stage = int(self._target_stage)
                                binding.vec_set_curriculum_target(self.c_envs, self._target_stage)

                            # Simple diagnostic print
                            dogfight_log(f'[CURRICULUM] step={total_steps} stage={self._target_stage:.2f} base={base_stage_perf:.3f}({self._base_stage_eps:.0f}eps) mastered={self._mastered_stage}')

                            # Base stage: decay by 10% each interval (so recent perf matters more)
                            self._base_stage_kills *= 0.9
                            self._base_stage_eps *= 0.9

        # Policy pool: periodic opponent swapping
        if self.policy_pool is not None and len(self.policy_pool) > 0:
            total_steps = self.tick * self.num_agents
            if total_steps - self._last_opponent_swap >= self.opponent_swap_interval:
                self._last_opponent_swap = total_steps
                new_opponent = self.policy_pool.select(
                    self._target_stage,
                    mode=self.opponent_selection
                )
                if new_opponent and new_opponent != self._current_opponent_path:
                    self._load_opponent_policy(new_opponent)
                    self._current_opponent_path = new_opponent

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

    def force_state(
        self,
        env_idx=0,
        player_pos=None,       # (x, y, z) tuple, default (0, 0, 1000)
        player_vel=None,       # (vx, vy, vz) tuple, default (150, 0, 0)
        player_ori=None,       # (w, x, y, z) quaternion, default (1, 0, 0, 0) = wings level
        player_throttle=1.0,   # [0, 1], default full throttle
        opponent_pos=None,     # (x, y, z) or None for auto (400m ahead)
        opponent_vel=None,     # (vx, vy, vz) or None for auto (match player)
        opponent_ori=None,     # (w, x, y, z) or None for auto (match player)
        tick=0,
        player_cooldown=None,  # Fire cooldown ticks for player (None = 0)
        opponent_cooldown=None, # Fire cooldown ticks for opponent (None = 0)
    ):
        """
        Force exact game state for testing/debugging.

        Usage:
            env.force_state(player_pos=(-1500, 0, 1000), player_vel=(150, 0, 0))
            env.force_state(player_vel=(80, 0, 0))  # Just change velocity
            env.force_state(player_cooldown=100, opponent_cooldown=100)  # Disable guns for 2 sec
        """
        # Build kwargs for C binding
        kwargs = {'tick': tick, 'p_throttle': player_throttle}

        # Player position
        if player_pos is not None:
            kwargs['p_px'], kwargs['p_py'], kwargs['p_pz'] = player_pos

        # Player velocity
        if player_vel is not None:
            kwargs['p_vx'], kwargs['p_vy'], kwargs['p_vz'] = player_vel

        # Player orientation
        if player_ori is not None:
            kwargs['p_ow'], kwargs['p_ox'], kwargs['p_oy'], kwargs['p_oz'] = player_ori

        # Opponent position (None = auto)
        if opponent_pos is not None:
            kwargs['o_px'], kwargs['o_py'], kwargs['o_pz'] = opponent_pos

        # Opponent velocity (None = auto)
        if opponent_vel is not None:
            kwargs['o_vx'], kwargs['o_vy'], kwargs['o_vz'] = opponent_vel

        # Opponent orientation (None = auto)
        if opponent_ori is not None:
            kwargs['o_ow'], kwargs['o_ox'], kwargs['o_oy'], kwargs['o_oz'] = opponent_ori

        # Fire cooldowns (None = 0, i.e., guns ready)
        if player_cooldown is not None:
            kwargs['p_cooldown'] = player_cooldown
        if opponent_cooldown is not None:
            kwargs['o_cooldown'] = opponent_cooldown

        # Call C binding with the specific env handle
        binding.env_force_state(self._env_handles[env_idx], **kwargs)

    def get_state(self, env_idx=0):
        """
        Get raw player state (independent of observation scheme).

        Returns dict with keys:
            px, py, pz: Position
            vx, vy, vz: Velocity
            ow, ox, oy, oz: Orientation quaternion
            up_x, up_y, up_z: Up vector (derived from quaternion)
            fwd_x, fwd_y, fwd_z: Forward vector (derived from quaternion)
            throttle: Current throttle

        Useful for physics tests that need exact state regardless of obs_scheme.
        """
        return binding.env_get_state(self._env_handles[env_idx])

    def set_autopilot(
        self,
        env_idx=0,
        mode=AutopilotMode.STRAIGHT,
        throttle=1.0,
        bank_deg=30.0,
        climb_rate=5.0,
    ):
        """
        Set autopilot mode for opponent aircraft.

        Args:
            env_idx: Environment index, or None for all environments
            mode: AutopilotMode constant (STRAIGHT, LEVEL, TURN_LEFT, etc.)
            throttle: Target throttle [0, 1]
            bank_deg: Bank angle for turn modes (degrees)
            climb_rate: Target vertical velocity for climb/descend (m/s)

        Usage:
            env.set_autopilot(mode=AutopilotMode.LEVEL)  # Level flight, env 0
            env.set_autopilot(mode=AutopilotMode.TURN_RIGHT, bank_deg=45)  # 45° right turn
            env.set_autopilot(mode=AutopilotMode.RANDOM)  # Randomize each episode
            env.set_autopilot(env_idx=None, mode=AutopilotMode.RANDOM)  # All envs
        """
        if env_idx is None:
            # Vectorized: set all envs at once
            binding.vec_set_autopilot(
                self.c_envs,
                mode=mode,
                throttle=throttle,
                bank_deg=bank_deg,
                climb_rate=climb_rate,
            )
        else:
            # Single env
            binding.env_set_autopilot(
                self._env_handles[env_idx],
                mode=mode,
                throttle=throttle,
                bank_deg=bank_deg,
                climb_rate=climb_rate,
            )

    def set_mode_weights(self, level=0.2, turn_left=0.2, turn_right=0.2,
                         climb=0.2, descend=0.2):
        """
        Set probability weights for AP_RANDOM mode selection.

        Weights should sum to 1.0. Used for curriculum learning to bias
        toward easier modes (e.g., LEVEL, STRAIGHT turns) early in training.

        Args:
            level: Weight for AP_LEVEL (maintain altitude)
            turn_left: Weight for AP_TURN_LEFT
            turn_right: Weight for AP_TURN_RIGHT
            climb: Weight for AP_CLIMB
            descend: Weight for AP_DESCEND
        """
        binding.vec_set_mode_weights(
            self.c_envs,
            level=level, turn_left=turn_left, turn_right=turn_right,
            climb=climb, descend=descend,
        )

    def get_autopilot_mode(self, env_idx=0):
        """Get current autopilot mode for an environment (for testing/debugging)."""
        return binding.env_get_autopilot_mode(self._env_handles[env_idx])

    def set_obs_highlight(self, indices, env_idx=0):
        """
        Set which observations to highlight with red arrows in the visual display.

        Args:
            indices: List of observation indices to highlight (e.g., [4, 5, 6] for pitch, roll, yaw)
            env_idx: Environment index

        Usage:
            env.set_obs_highlight([4, 5, 6])  # Highlight pitch, roll, yaw in scheme 0
            env.set_obs_highlight([])  # Clear highlights
        """
        binding.env_set_obs_highlight(self._env_handles[env_idx], list(indices))

    def set_curriculum_stage(self, stage: int):
        """
        Set curriculum stage for all environments (global curriculum).

        Called by training loop based on aggregate kill_rate from log data.
        All envs share the same stage for coherent metrics.

        Args:
            stage: Curriculum stage (0=TAIL_CHASE, 1=HEAD_ON, 2=VERTICAL,
                   3=MANEUVERING, 4=OFFSET_MANEUVERING, 5=ANGLED_MANEUVERING,
                   6=FULL_RANDOM, 7=HARD_MANEUVERING, 8=CROSSING, 9=EVASIVE)
        """
        binding.vec_set_curriculum_stage(self.c_envs, stage)
        self._current_stage = stage

    def get_curriculum_stage(self) -> int:
        """Get current global curriculum stage."""
        return self._current_stage

    def set_curriculum_target(self, target: float):
        """
        Set curriculum target (0.0-15.0) for probabilistic stage assignment.

        At each episode reset, stage is assigned probabilistically:
        - target=1.3 → 70% stage 1, 30% stage 2

        Args:
            target: Float target from 0.0 to 17.0 (18 curriculum stages)
        """
        self._target_stage = max(0.0, min(target, 17.0))
        binding.vec_set_curriculum_target(self.c_envs, self._target_stage)
        self._current_stage = int(self._target_stage)

    def get_curriculum_target(self) -> float:
        """Get current curriculum target (float 0.0-9.0)."""
        return self._target_stage

    def get_autoace_state(self, env_idx=0):
        """
        Get AutoAce opponent state and tactical info for behavioral tests.

        Returns dict with keys:
            # Opponent plane state
            opp_px, opp_py, opp_pz: Position
            opp_vx, opp_vy, opp_vz: Velocity
            opp_fwd_x, opp_fwd_y, opp_fwd_z: Forward vector
            opp_ow, opp_ox, opp_oy, opp_oz: Orientation quaternion

            # Last AutoAce actions
            opp_throttle, opp_elevator, opp_aileron, opp_rudder, opp_trigger

            # Tactical state
            engagement: 0=OFFENSIVE, 1=NEUTRAL, 2=DEFENSIVE, 3=WEAPONS, 4=EXTEND
            mode: Autopilot mode enum value
            aspect_angle: Degrees (0=behind target, 180=head-on)
            antenna_train: Target bearing from nose (0=dead ahead)
            range: Distance in meters
            closure_rate: Positive = closing (m/s)
            in_gun_envelope: Boolean
        """
        return binding.env_get_autoace_state(self._env_handles[env_idx])

    def set_camera_follow(self, follow_opponent=False, env_idx=0):
        """
        Set which plane the camera follows during rendering.

        Args:
            follow_opponent: True to follow opponent (AutoAce), False to follow player
            env_idx: Environment index
        """
        binding.env_set_camera_follow(self._env_handles[env_idx], 1 if follow_opponent else 0)

    def set_eval_spawn_mode(self, mode: int):
        """
        Set eval spawn mode for all environments.

        Args:
            mode: 0 = random (default), 1 = opponent_advantage, 2 = symmetric merge

        Mode 1 (opponent_advantage) places opponent 400m behind player at 15° off tail,
        giving opponent an easy kill opportunity. Useful for testing if opponent can kill.

        Mode 2 (symmetric scenario pool) randomly selects from 3 scenarios:
        - Head-on merge: facing each other, guns locked until pass
        - Post-merge zoom: both nose-up climbing away, tests energy management
        - Turning fight: both banked and pulling, tests turn performance
        All scenarios are symmetric with slight perturbations (±5m pos, ±2° heading,
        ±3 m/s speed) to break identical observations. Used for fair Elo evaluation.
        """
        binding.vec_set_eval_spawn_mode(self.c_envs, mode)

    def set_flight_params(
        self,
        env_idx=None,
        control_v_ref=None,
        control_scale_slope=None,
        control_scale_min=None,
        damping_scale_slope=None,
        damping_multiplier=None,
    ):
        """
        Set flight physics parameters for parameter sweeps.

        Args:
            env_idx: Environment index, or None for all environments
            control_v_ref: Reference speed for full control authority (default 100.0 m/s)
            control_scale_slope: Authority reduction per m/s above ref (default 0.000833)
            control_scale_min: Minimum authority floor (default 0.05 = 5%)
            damping_scale_slope: Extra damping per m/s above ref (default 0.0)
            damping_multiplier: Scale all damping (CM_Q, CL_P, CN_R). 1.0=normal, 2.0=double

        Usage:
            # Increase damping for smoother recovery
            env.set_flight_params(damping_multiplier=2.0)

            # Sweep parameter combinations
            env.set_flight_params(damping_multiplier=1.5, control_scale_min=0.10)
        """
        kwargs = {}
        if control_v_ref is not None:
            kwargs['control_v_ref'] = control_v_ref
        if control_scale_slope is not None:
            kwargs['control_scale_slope'] = control_scale_slope
        if control_scale_min is not None:
            kwargs['control_scale_min'] = control_scale_min
        if damping_scale_slope is not None:
            kwargs['damping_scale_slope'] = damping_scale_slope
        if damping_multiplier is not None:
            kwargs['damping_multiplier'] = damping_multiplier

        if env_idx is None:
            binding.vec_set_flight_params(self.c_envs, **kwargs)
        else:
            binding.env_set_flight_params(self._env_handles[env_idx], **kwargs)


def test_performance(timeout=10, atn_cache=1024):
    env = Dogfight(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")


if __name__ == "__main__":
    test_performance()
