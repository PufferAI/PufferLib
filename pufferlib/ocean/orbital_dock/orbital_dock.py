'''Orbital rendezvous and docking environment.

A 3D orbital mechanics environment where the agent controls a chaser spacecraft
that must navigate through space, rendezvous with, and dock to a space station
in orbit.

Implements hierarchical velocity control (Hovell & Ulrich, 2021):
- Policy outputs DESIRED VELOCITY, not thrust
- P controller converts velocity command to thrust
- This separates guidance (learned) from control (engineered)
- Makes the learning problem much easier

Key features:
- Full 3D orbital mechanics with gravity
- LVLH (Local Vertical Local Horizontal) reference frame
- Continuous action space for 3-axis velocity commands
- P controller handles thrust generation
'''

import gymnasium
import numpy as np
import torch
import torch.nn as nn

import pufferlib
import pufferlib.models
import pufferlib.pytorch
from pufferlib.ocean.orbital_dock import binding


class Policy(pufferlib.models.Default):
    '''Custom policy with lower initial action std for precise velocity control.'''
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        if self.is_continuous:
            # Start with std=0.37 instead of 1.0
            # More precise initial exploration for docking
            nn.init.constant_(self.decoder_logstd, -1.0)


class OrbitalDock(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        report_interval=128,
        buf=None,
        seed=0,
        # Physics (defaults match config/ocean/orbital_dock.ini)
        mu=3.986e14,
        station_radius=6.771e6,
        dt=1.0,
        max_thrust=5000.0,
        mass=10000.0,
        fuel_budget=100.0,
        max_steps=500,
        # Hierarchical velocity control (Hovell & Ulrich 2021)
        kp=0.5,                      # P controller gain
        max_cmd_vel=2.0,             # Max commanded velocity (m/s)
        # Docking conditions
        dock_dist=5.0,
        dock_speed=0.5,
        # Difficulty (unused with hierarchical control)
        difficulty=0.0,
        # Reward weights
        reward_dock=100.0,           # Terminal dock bonus
        reward_dist_shaping=10.0,    # Distance progress scale
        reward_closing=0.0,          # Unused
        reward_vel_match=0.5,        # Velocity penalty near dock
        reward_fuel_penalty=0.0,     # Disabled for now
        reward_crash=50.0,           # Crash penalty
        reward_deorbit=50.0,
        reward_escape=50.0,
        reward_plane_align=0.0,
        reward_node_timing=0.0,
    ):
        self.kp = kp
        self.max_cmd_vel = max_cmd_vel

        # 14-dimensional observation space
        # [rel_r, rel_v, rel_h, rel_vr, rel_vv, rel_vh, dist_norm, closing_speed,
        #  fuel_remaining, orbit_alt_norm, phase_angle, inclination_diff,
        #  node_angle, time_remaining]
        # Normalized to approximately [-1, 1] using pos_scale=100m, vel_scale=2m/s
        # Position obs can exceed [-1,1] at longer distances, so use [-10,10] bounds
        self.single_observation_space = gymnasium.spaces.Box(
            low=-10.0, high=10.0, shape=(14,), dtype=np.float32
        )

        # Continuous action space: desired velocity in LVLH frame
        # [vel_r, vel_v, vel_h] in m/s
        # Policy outputs desired velocity, P controller converts to thrust
        self.single_action_space = gymnasium.spaces.Box(
            low=-max_cmd_vel, high=max_cmd_vel, shape=(3,), dtype=np.float32
        )

        self.render_mode = None if render_mode in (None, 'None') else render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)

        # Initialize C environments
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            # Physics
            mu=mu,
            station_radius=station_radius,
            dt=dt,
            max_thrust=max_thrust,
            mass=mass,
            fuel_budget=fuel_budget,
            max_steps=max_steps,
            # Hierarchical velocity control
            kp=kp,
            max_cmd_vel=max_cmd_vel,
            # Docking conditions
            dock_dist=dock_dist,
            dock_speed=dock_speed,
            # Difficulty (unused)
            difficulty=difficulty,
            # Reward weights
            reward_dock=reward_dock,
            reward_dist_shaping=reward_dist_shaping,
            reward_closing=reward_closing,
            reward_vel_match=reward_vel_match,
            reward_fuel_penalty=reward_fuel_penalty,
            reward_crash=reward_crash,
            reward_deorbit=reward_deorbit,
            reward_escape=reward_escape,
            reward_plane_align=reward_plane_align,
            reward_node_timing=reward_node_timing,
        )

    def reset(self, seed=0):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        # Actions are continuous velocity commands [vel_r, vel_v, vel_h]
        # The P controller in C code converts these to thrust
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        # Auto-render when render_mode is 'human'
        if self.render_mode == 'human':
            self.render()

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024, num_envs=4096):
    '''Test environment performance (steps per second).'''
    import time

    env = OrbitalDock(num_envs=num_envs)
    env.reset()
    tick = 0

    # Pre-generate random velocity commands: shape (cache_size, num_envs, 3)
    # Range [-max_cmd_vel, max_cmd_vel]
    actions = np.random.uniform(-2.0, 2.0, (atn_cache, num_envs, 3)).astype(np.float32)

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    elapsed = time.time() - start
    sps = int(num_envs * tick / elapsed)
    print(f'OrbitalDock SPS: {sps:,}')
    env.close()
    return sps


def test_basic():
    '''Basic functionality test.'''
    print('Testing OrbitalDock basic functionality...')

    env = OrbitalDock(num_envs=2)
    obs, _ = env.reset(seed=42)
    print(f'  Observation shape: {obs.shape}')
    print(f'  Observation range: [{obs.min():.3f}, {obs.max():.3f}]')

    # Take a few steps with zero velocity command (coast)
    zero_vel = np.zeros((2, 3), dtype=np.float32)
    for i in range(10):
        obs, rewards, terminals, truncations, info = env.step(zero_vel)

    print(f'  After 10 steps (zero velocity command):')
    print(f'    Rewards: {rewards}')
    print(f'    Terminals: {terminals}')

    # Take steps with random velocity commands
    for i in range(100):
        actions = np.random.uniform(-2.0, 2.0, (2, 3)).astype(np.float32)
        obs, rewards, terminals, truncations, info = env.step(actions)

    print(f'  After 100 random steps:')
    print(f'    Observation range: [{obs.min():.3f}, {obs.max():.3f}]')

    env.close()
    print('  Basic test passed!')


if __name__ == '__main__':
    test_basic()
    print()
    test_performance()
