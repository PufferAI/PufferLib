'''
High-perf Boids
Inspired by https://people.ece.cornell.edu/land/courses/ece4760/labs/s2021/Boids/Boids.html
'''

from code import interact
import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.boids.cy_boids import CyBoids

class Boids(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=2,
        width=500,
        height=640,
        num_boids=1,
        buf=None,
        render_mode=None,
        report_interval=128
    ):
        self.num_agents = num_envs
        self.single_action_space = gymnasium.spaces.Box(-3.0, 3.0, shape=(2,))
        self.single_observation_space = gymnasium.spaces.Box(-1000.0, 1000.0, shape=(num_boids, 2))
        self.render_mode = render_mode
        self.report_interval = report_interval

        super().__init__(buf)
        self.actions = self.actions.flatten()
        self.c_envs = CyBoids(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            num_envs,
            num_boids,
        )
 
    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        self.actions[:] = np.clip(actions.flatten(), -1, 1)
        print("PYTHON ACTIONS", self.actions)
        self.c_envs.step()

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, [])

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    env = Boids(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start)}')


if __name__ == '__main__':
    test_performance()
