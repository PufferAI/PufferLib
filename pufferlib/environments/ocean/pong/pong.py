'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

from raylib import rl

import pufferlib
from pufferlib.environments.ocean.pong.cy_pong import CyPong

class MyPong(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None,
            width=500, height=640, paddle_width=20, paddle_height=70,
            ball_width=10, ball_height=15, paddle_speed=8,
            ball_initial_speed_x=10, ball_initial_speed_y=1,
            ball_speed_y_increment=3, ball_max_speed_y=13,
            max_score=21):
        super().__init__()

        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode

        # sim hparams (px, px/tick)
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.paddle_speed = paddle_speed
        self.ball_initial_speed_x = ball_initial_speed_x
        self.ball_initial_speed_y = ball_initial_speed_y
        self.ball_speed_y_increment = ball_speed_y_increment
        self.ball_max_speed_y = ball_max_speed_y
        self.max_score = max_score

        # sim data (coordinates are bottom-left increasing towards top-right)
        self.paddle_yl_yr = np.zeros((self.num_envs, 2,), dtype=np.float32)
        self.ball_x_y = np.zeros((self.num_envs, 2,), dtype=np.float32)
        self.ball_vx_vy = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.score_l_r = np.zeros((self.num_envs, 2,), dtype=np.uint32)
        self.misc_logging = np.zeros((self.num_envs, 10,), dtype=np.uint32)

        # spaces
        self.num_obs = 8
        self.num_act = 3
        self.observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_observation_space = self.observation_space
        self.action_space = gymnasium.spaces.Discrete(self.num_act)
        self.single_action_space = self.action_space
        self.human_action = None

        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations = np.zeros((self.num_agents, self.num_obs,), dtype=np.float32),
            rewards = np.zeros(self.num_agents, dtype=np.float32),
            terminals = np.zeros(self.num_agents, dtype=np.bool),
            truncations = np.zeros(self.num_agents, dtype=bool),
            masks = np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint32)
        self.terminals_uint8 = np.zeros(self.num_agents, dtype=np.uint8)

        self.tick = 0
        self.report_interval = 100
        self.reward_sum = 0
        self.reward_sum = 0
        self.num_finished_games = 0
        self.wins_sum = 0
        self.n_bounces_sum = 0
        self.ticks_sum = 0

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs = []
        for i in range(self.num_envs):
            # TODO: since single agent, could we just pass values by reference instead of (1,) array?
            self.c_envs.append(CyPong(
                self.buf.observations[i], self.actions[i:i+1],
                self.buf.rewards[i:i+1], self.terminals_uint8[i:i+1],
                self.paddle_yl_yr[i], self.ball_x_y[i], self.ball_vx_vy[i],
                self.score_l_r[i], self.width, self.height,
                self.paddle_width, self.paddle_height, self.ball_width, self.ball_height,
                self.paddle_speed, self.ball_initial_speed_x, self.ball_initial_speed_y,
                self.ball_max_speed_y, self.ball_speed_y_increment, self.max_score,
                self.misc_logging[i],))
            self.c_envs[i].reset()

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        for i in range(self.num_envs):
            self.c_envs[i].step()

        # TODO: hacky way to convert uint8 to bool
        self.buf.terminals[:] = self.terminals_uint8.astype(bool)

        # self.misc_logging[0] = 1  # bool: round is over, log
        # self.misc_logging[1] = self.tick
        # self.misc_logging[2] = self.n_bounces
        # self.misc_logging[3] = self.win

        info = {}
        self.reward_sum += self.buf.rewards.mean()
        finished_rounds_mask = self.misc_logging[:,0] == 1
        self.num_finished_games += np.sum(finished_rounds_mask)
        self.ticks_sum += self.misc_logging[finished_rounds_mask, 1].sum()
        self.n_bounces_sum += self.misc_logging[finished_rounds_mask, 2].sum()
        self.wins_sum += self.misc_logging[finished_rounds_mask, 3].sum()
        if self.tick % self.report_interval == 0:
            win_rate = self.wins_sum / self.num_finished_games if self.num_finished_games > 0 else 0
            info.update({
                'reward': self.reward_sum / self.report_interval,
                'num_games': self.num_finished_games,
                'num_wins': self.wins_sum,
                'winrate': win_rate,
                'bounces_per_game': self.n_bounces_sum / self.num_finished_games if self.num_finished_games > 0 else 0,
                'ticks_per_game': self.ticks_sum / self.num_finished_games if self.num_finished_games > 0 else 0,
            })
            self.reward_sum = 0
            self.num_finished_games = 0
            self.wins_sum = 0
            self.n_bounces_sum = 0
            self.ticks_sum = 0

        self.tick += 1

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        self.c_envs[0].render()

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
