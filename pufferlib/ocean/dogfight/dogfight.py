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


class Dogfight(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        render_mode=None,
        report_interval=1,
        buf=None,
        seed=42,
        max_steps=3000,
    ):
        # player(13) + rel_pos(3) + rel_vel(3) = 19
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(19,),
            dtype=np.float32,
        )

        # Action: Box(5) continuous [-1, 1]
        # [0] throttle, [1] elevator, [2] ailerons, [3] rudder, [4] trigger
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32
        )

        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

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
                report_interval=self.report_interval,
                max_steps=max_steps,
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
