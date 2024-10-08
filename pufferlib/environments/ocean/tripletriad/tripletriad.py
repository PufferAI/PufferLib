import numpy as np
import gymnasium

import pufferlib
from pufferlib.environments.ocean.tripletriad.cy_tripletriad import CyTripleTriad

class MyTripleTriad(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            width=990, height=1000, piece_width=192, piece_height=224, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(114,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(15)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf=buf)
        self.c_envs = CyTripleTriad(self.observations, self.actions,
            self.rewards, self.terminals, num_envs, width, height,
            piece_width, piece_height)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

def test_performance(timeout=10, atn_cache=1024):
    env = MyTripleTriad(num_envs=1000)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 2, (atn_cache, env.num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_envs * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
