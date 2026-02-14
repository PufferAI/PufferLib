'''Orbital rendezvous and docking environment (STELLAR / Chen et al. AAS 2023).

CW linear relative motion dynamics in LVLH frame.
Direct thrust control with PPO.
Reproduces the STELLAR implementation in PufferLib.
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
        # Physics
        mu=3.986e14,
        station_radius=42164000.0,
        dt=1.0,
        max_thrust=10.0,
        mass=500.0,
        fuel_budget=100.0,
        max_steps=2500,
        # Docking point (STELLAR: [0, 60, 0] in LVLH)
        dock_x=0.0,
        dock_y=60.0,
        dock_z=0.0,
        dock_dist=10.0,
        dock_speed=2.0,
        dock_speed_start=10.0,
        anneal_steps=200000,
        # LOS cone
        los_angle=60.0,
        los_extent=800.0,
        # Initial conditions (STELLAR V-bar approach)
        init_x_center=0.0,
        init_y_center=800.0,
        init_z_center=0.0,
        init_x_range=400.0,
        init_y_range=300.0,
        init_z_range=400.0,
    ):
        # 10-dimensional observation: raw LVLH state + computed features
        # [x, y, z, vx, vy, vz, dist, speed, closing_vel, time_remaining]
        self.single_observation_space = gymnasium.spaces.Box(
            low=-2000.0, high=2000.0, shape=(10,), dtype=np.float32
        )

        # Continuous action space: normalized thrust fractions in LVLH frame
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.render_mode = None if render_mode in (None, 'None') else render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            mu=mu,
            station_radius=station_radius,
            dt=dt,
            max_thrust=max_thrust,
            mass=mass,
            fuel_budget=fuel_budget,
            max_steps=max_steps,
            dock_x=dock_x,
            dock_y=dock_y,
            dock_z=dock_z,
            dock_dist=dock_dist,
            dock_speed=dock_speed,
            dock_speed_start=dock_speed_start,
            anneal_steps=anneal_steps,
            los_angle=los_angle,
            los_extent=los_extent,
            init_x_center=init_x_center,
            init_y_center=init_y_center,
            init_z_center=init_z_center,
            init_x_range=init_x_range,
            init_y_range=init_y_range,
            init_z_range=init_z_range,
        )

    def reset(self, seed=0):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

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
    '''Test environment performance.'''
    import time
    env = OrbitalDock(num_envs=num_envs)
    env.reset()
    tick = 0
    actions = np.random.uniform(-1.0, 1.0, (atn_cache, num_envs, 3)).astype(np.float32)
    start = time.time()
    while time.time() - start < timeout:
        env.step(actions[tick % atn_cache])
        tick += 1
    elapsed = time.time() - start
    sps = int(num_envs * tick / elapsed)
    print(f'OrbitalDock SPS: {sps:,}')
    env.close()
    return sps


def test_basic():
    '''Basic functionality test.'''
    print('Testing OrbitalDock (CW dynamics)...')
    env = OrbitalDock(num_envs=2)
    obs, _ = env.reset(seed=42)
    print(f'  Obs shape: {obs.shape}')
    print(f'  Obs range: [{obs.min():.1f}, {obs.max():.1f}]')
    print(f'  Obs[0]: x={obs[0,0]:.1f} y={obs[0,1]:.1f} z={obs[0,2]:.1f} '
          f'vx={obs[0,3]:.3f} vy={obs[0,4]:.3f} vz={obs[0,5]:.3f}')

    # Coast (zero thrust)
    zero = np.zeros((2, 3), dtype=np.float32)
    for i in range(10):
        obs, r, t, tr, info = env.step(zero)
    print(f'  After 10 steps (zero thrust):')
    print(f'    Rewards: {r}')
    print(f'    Obs[0]: x={obs[0,0]:.1f} y={obs[0,1]:.1f} z={obs[0,2]:.1f}')

    # Random thrust
    for i in range(100):
        actions = np.random.uniform(-1.0, 1.0, (2, 3)).astype(np.float32)
        obs, r, t, tr, info = env.step(actions)
    print(f'  After 100 random steps:')
    print(f'    Obs range: [{obs.min():.1f}, {obs.max():.1f}]')

    env.close()
    print('  Test passed!')


if __name__ == '__main__':
    test_basic()
    print()
    test_performance()
