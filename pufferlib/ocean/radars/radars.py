"""A simple sample environment. Use this as a template for your own envs."""

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.radars.cy_radars import CyRadars

# Time is always milliseconds.
# Distance is always millimeters.
# Velocities are millimeters per millisecond, equal to meter per second (m/s).

MAX_AZ_SLICES = 30
MAX_EL_SLICES = 10

MAX_SEARCHERS = 1
FEATURES_PER_TRACKER = 3

PLACEHOLDER_FOR_SENSOR_ID = 1

MAX_EARLY = 30000  # 30 seconds
MAX_TARDY = -30000  # -30 seconds


class Radars(pufferlib.PufferEnv):
    def __init__(
        self, num_envs=1, render_mode=None, buf=None, initial_targets=5, max_trackers=5
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=MAX_TARDY,
            high=MAX_EARLY,
            shape=(
                MAX_AZ_SLICES * MAX_EL_SLICES
                + max_trackers * FEATURES_PER_TRACKER
                + PLACEHOLDER_FOR_SENSOR_ID,
            ),
            dtype=np.int16,
        )
        self.single_action_space = gymnasium.spaces.Discrete(
            MAX_SEARCHERS + max_trackers
        )
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.max_trackers = max_trackers

        super().__init__(buf)
        self.c_envs = CyRadars(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            num_envs,
            initial_targets,
            max_trackers,
        )

    def reset(self, seed=None):
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()

        episode_returns = self.rewards[self.terminals]

        info = []
        if len(episode_returns) > 0:
            info = [
                {
                    "reward": np.mean(episode_returns),
                }
            ]

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()


def test_performance(timeout=100):
    env = Radars(max_trackers=300, initial_targets=300)
    env.reset()
    tick = 0

    import time

    start = time.time()
    while time.time() - start < timeout:
        env.step(0)
        env.render()
        tick += 1
        env.step(np.random.randint(0, 1 + env.max_trackers, size=env.num_agents))
        env.render()
        tick += 1

    print(f"SPS: {tick / (time.time() - start)}")


if __name__ == "__main__":
    test_performance()
