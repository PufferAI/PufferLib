'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.environments.ocean.pong.cy_pong import CyPong

class MyPong(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
            width=500, height=640, paddle_width=20, paddle_height=70,
            ball_width=32, ball_height=32, paddle_speed=8,
            ball_initial_speed_x=10, ball_initial_speed_y=1,
            ball_speed_y_increment=3, ball_max_speed_y=13,
            max_score=21, frameskip=1, report_interval=128, buf=None):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(8,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(3)
        self.render_mode = render_mode
        self.num_agents = num_envs

        self.report_interval = report_interval
        self.human_action = None
        self.tick = 0

        super().__init__(buf)
        self.c_envs = CyPong(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, width, height,
            paddle_width, paddle_height, ball_width, ball_height,
            paddle_speed, ball_initial_speed_x, ball_initial_speed_y,
            ball_max_speed_y, ball_speed_y_increment, max_score, frameskip)
 
    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
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
    env = MyPong(num_envs=1000)
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
