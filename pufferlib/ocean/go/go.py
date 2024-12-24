'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.go.cy_go import CyGo

class Go(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            width=950, height=800,
            grid_size=7,
            board_width=600, board_height=600,
            grid_square_size=600/9,
            moves_made=0,
            komi=7.5,
            score = 0.0,
            last_capture_position=-1,
            reward_move_pass = -0.25,
            reward_move_invalid = -0.1,
            reward_move_valid = 0.1,
            reward_player_capture = 0.25,
            reward_opponent_capture = -0.25,
            buf = None):

        # env
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        self.num_obs = (grid_size) * (grid_size)*2 + 2
        self.num_act = (grid_size) * (grid_size) + 1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(self.num_act)

        super().__init__(buf=buf)
        height = 64*(grid_size+1)
        self.c_envs = CyGo(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, width, height, grid_size, board_width,
            board_height, grid_square_size, moves_made, komi, score,
            last_capture_position, reward_move_pass, reward_move_invalid,
            reward_move_valid, reward_player_capture, reward_opponent_capture)

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

    def close(self):
        self.c_envs.close()

def test_performance(timeout=10, atn_cache=1024):
    num_envs=1000;
    env = Go(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')
if __name__ == '__main__':
    test_performance()
