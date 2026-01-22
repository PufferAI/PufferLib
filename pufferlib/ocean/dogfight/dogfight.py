import time
import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.dogfight import binding


# Autopilot mode constants (must match autopilot.h enum)
class AutopilotMode:
    STRAIGHT = 0     # Fly straight (current/default behavior)
    LEVEL = 1        # Level flight with PD on vz
    TURN_LEFT = 2    # Coordinated left turn
    TURN_RIGHT = 3   # Coordinated right turn
    CLIMB = 4        # Constant climb rate
    DESCEND = 5      # Constant descent rate
    RANDOM = 6       # Random mode selection at reset


# Observation sizes by scheme (must match C OBS_SIZES in dogfight.h)
OBS_SIZES = {
    0: 12,  # ANGLES: pos(3) + speed(1) + euler(3) + target_angles(4) + opp(1)
    1: 13,  # PURSUIT: speed(1) + pot(1) + euler(2) + energy(1) + target(4) + tgt_state(3) + energy_adv(1)
    2: 10,  # REALISTIC: instruments(4) + gunsight(3) + visual(3)
    3: 10,  # REALISTIC_RANGE: instruments(4) + gunsight(3) + visual(3) w/ km range
    4: 13,  # REALISTIC_ENEMY_STATE: + enemy pitch/roll/heading
    5: 15,  # REALISTIC_FULL: + turn rate + G-loading
}


class Dogfight(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        render_mode=None,
        render_fps=None,  # Target FPS when rendering (None=no delay, 50=real-time, 10=slow-mo)
        report_interval=1,
        buf=None,
        seed=42,
        max_steps=3000,
        obs_scheme=0,
        # Curriculum learning
        curriculum_enabled=0,       # 0=off (legacy), 1=on (progressive stages)
        curriculum_randomize=0,     # 0=progressive (training), 1=random stage each episode (eval)
        advance_threshold=0.7,
        demote_threshold=0.3,
        eval_window=50,
        # df11: Simplified rewards (6 terms)
        reward_aim_scale=0.05,       # Continuous aiming reward
        reward_closing_scale=0.003,  # Per m/s closing
        penalty_neg_g=0.02,          # Enforce "pull to turn"
        penalty_stall=0.002,         # Speed safety
        penalty_rudder=0.001,        # Prevent knife-edge
        speed_min=50.0,              # Stall threshold
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

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)  # REQUIRED for continuous

        # Print hyperparameters at init (for sweep debugging)
        print(f"=== DOGFIGHT ENV INIT ===")
        print(f"  obs_scheme={obs_scheme}, num_envs={num_envs}")
        print(f"  REWARDS: aim={reward_aim_scale:.4f} closing={reward_closing_scale:.4f}")
        print(f"  PENALTY: neg_g={penalty_neg_g:.4f} stall={penalty_stall:.4f} rudder={penalty_rudder:.4f}")
        print(f"  curriculum={curriculum_enabled}, advance={advance_threshold}, demote={demote_threshold}")

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
                advance_threshold=advance_threshold,
                demote_threshold=demote_threshold,
                eval_window=eval_window,

                reward_aim_scale=reward_aim_scale,
                reward_closing_scale=reward_closing_scale,
                penalty_neg_g=penalty_neg_g,
                penalty_stall=penalty_stall,
                penalty_rudder=penalty_rudder,
                speed_min=speed_min,
            )
            self._env_handles.append(handle)

        self.c_envs = binding.vectorize(*self._env_handles)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed else 0)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
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
    ):
        """
        Force exact game state for testing/debugging.

        Usage:
            env.force_state(player_pos=(-1500, 0, 1000), player_vel=(150, 0, 0))
            env.force_state(player_vel=(80, 0, 0))  # Just change velocity
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
