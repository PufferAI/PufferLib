'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.rocket_lander.cy_rocket_lander import CyRocketLander

class RocketLander(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=32, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(6,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval

        super().__init__(buf)
        self.float_actions = np.zeros((num_envs, 3), dtype=np.float32)
        self.c_envs = CyRocketLander(self.observations, self.float_actions, self.rewards,
            self.terminals, self.truncations, num_envs)

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        self.float_actions[:, :] = 0
        self.float_actions[:, 0] = actions == 1
        self.float_actions[:, 1] = actions == 2
        self.float_actions[:, 2] = actions == 3
        self.c_envs.step()

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                info.append(log)

        self.tick += 1
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    env = RocketLander(num_envs=1000)
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
