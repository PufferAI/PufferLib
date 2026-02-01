'''Orbital rendezvous and docking environment.

A 3D orbital mechanics environment where the agent controls a chaser spacecraft
that must navigate through space, rendezvous with, and dock to a space station
in orbit. The challenge is learning counterintuitive orbital mechanics: to catch
a target ahead of you, you must thrust retrograde to drop into a lower, faster orbit.

Key features:
- Full 3D orbital mechanics with gravity
- LVLH (Local Vertical Local Horizontal) reference frame for observations/actions
- Multi-discrete action space for 3-axis thrust control
- Plane changes at ascending/descending nodes
- Fuel-optimal maneuver learning
'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.orbital_dock import binding


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
        max_thrust=500.0,
        mass=10000.0,
        fuel_budget=100.0,
        max_steps=2000,
        # Docking conditions
        dock_dist=5.0,
        dock_speed=0.5,
        # Difficulty (0.0 = simple starting conditions)
        difficulty=0.0,
        # Reward weights
        reward_dock=10.0,
        reward_dist_shaping=0.01,
        reward_closing=0.1,
        reward_vel_match=1.0,
        reward_fuel_penalty=0.01,
        reward_crash=5.0,
        reward_deorbit=5.0,
        reward_escape=5.0,
        reward_plane_align=0.0,
        reward_node_timing=0.0,
    ):
        # 14-dimensional observation space
        # [rel_x, rel_y, rel_z, rel_vx, rel_vy, rel_vz, dist_norm, closing_speed,
        #  fuel_remaining, orbit_alt_norm, phase_angle, inclination_diff,
        #  node_angle, time_remaining]
        # Normalized to approximately [-1, 1] using pos_scale=100m, vel_scale=2m/s
        # Position/velocity obs can exceed [-1,1] if agent drifts far, so use [-5,5] bounds
        self.single_observation_space = gymnasium.spaces.Box(
            low=-5.0, high=5.0, shape=(14,), dtype=np.float32
        )

        # Multi-discrete action space: 5x5x5 = 125 actions
        # [thrust_prograde, thrust_radial, thrust_normal]
        # Each dimension: {0: -100%, 1: -50%, 2: 0%, 3: +50%, 4: +100%}
        self.single_action_space = gymnasium.spaces.MultiDiscrete([5, 5, 5])

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
            # Docking conditions
            dock_dist=dock_dist,
            dock_speed=dock_speed,
            # Difficulty
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

    # Pre-generate random actions: shape (cache_size, num_envs, 3)
    actions = np.random.randint(0, 5, (atn_cache, num_envs, 3), dtype=np.int32)

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

    env = OrbitalDock(num_envs=2, difficulty=0.0)
    obs, _ = env.reset(seed=42)
    print(f'  Observation shape: {obs.shape}')
    print(f'  Observation range: [{obs.min():.3f}, {obs.max():.3f}]')

    # Take a few steps with no thrust (action=2 is 0%)
    no_thrust = np.full((2, 3), 2, dtype=np.int32)
    for i in range(10):
        obs, rewards, terminals, truncations, info = env.step(no_thrust)

    print(f'  After 10 steps (no thrust):')
    print(f'    Rewards: {rewards}')
    print(f'    Terminals: {terminals}')

    # Take steps with random actions
    for i in range(100):
        actions = np.random.randint(0, 5, (2, 3), dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(actions)

    print(f'  After 100 random steps:')
    print(f'    Observation range: [{obs.min():.3f}, {obs.max():.3f}]')

    env.close()
    print('  Basic test passed!')


if __name__ == '__main__':
    test_basic()
    print()
    test_performance()
