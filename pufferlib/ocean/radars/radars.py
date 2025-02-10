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
MAX_TRACKERS = 5
FEATURES_PER_TRACKER = 3

PLACEHOLDER_FOR_SENSOR_ID = 1

MAX_EARLY = 30000 # 30 seconds
MAX_TARDY = -30000 # -30 seconds


class Radars(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, buf=None, initial_targets=5):
        self.single_observation_space = gymnasium.spaces.Box(
            low=MAX_TARDY, 
            high=MAX_EARLY,
            shape=(
                MAX_AZ_SLICES * MAX_EL_SLICES
                + MAX_TRACKERS * FEATURES_PER_TRACKER
                + PLACEHOLDER_FOR_SENSOR_ID,
            ),
            dtype=np.int16,
        )
        self.single_action_space = gymnasium.spaces.Discrete(
            MAX_SEARCHERS + MAX_TRACKERS
        )
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = CyRadars(
            self.observations, self.actions, self.rewards, self.terminals, num_envs, initial_targets
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


def test_performance(timeout=10):
    env = Radars()
    env.reset()
    tick = 0

    actions = np.random.randint(0, MAX_SEARCHERS + MAX_TRACKERS, size=env.num_agents)

    import time

    start = time.time()
    while time.time() - start < timeout:
        env.step(actions)
        tick += 1

    print(f"SPS: %f", tick / (time.time() - start))


if __name__ == "__main__":
    test_performance()
