'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

from raylib import rl

import pufferlib
from pufferlib.environments.ocean.my_pong.c_my_pong import CMyPong, step_all
from pufferlib.environments.ocean import render

class MyPong(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None):
        super().__init__()

        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode

        # sim hparams (px, px/tick)
        self.width = 500
        self.height = 640
        self.paddle_width = 20
        self.paddle_height = 70
        self.ball_width = 10
        self.ball_height = 15
        self.paddle_speed = 8
        self.ball_initial_speed_x = 10
        self.ball_initial_speed_y = 1
        self.ball_speed_y_increment = 3
        self.ball_max_speed_y = 13
        self.max_score = 1  # 21

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

        if render_mode == 'human':
            self.client = RaylibClient(self.width, self.height,
                self.paddle_width, self.paddle_height, self.ball_width, self.ball_height)
        elif render_mode is None:
            pass
        else:
            raise ValueError(f'Invalid render mode: {render_mode}')

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs = []
        for i in range(self.num_envs):
            # TODO: since single agent, could we just pass values by reference instead of (1,) array?
            self.c_envs.append(CMyPong(
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
        if self.render_mode == 'human' and self.human_action is not None:
            self.actions[0] = self.human_action

        step_all(self.c_envs)

        # TODO: hacky way to convert uint8 to bool
        self.buf.terminals[:] = self.terminals_uint8.astype(np.bool)


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
                'win_rate': win_rate,
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
        if self.render_mode == 'human':
            self.human_action = None  # still
            if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_RIGHT):
                self.human_action = 1  # up
            elif rl.IsKeyDown(rl.KEY_DOWN) or rl.IsKeyDown(rl.KEY_LEFT):
                self.human_action = 2  # down
            return self.client.render(self.paddle_yl_yr[0], self.ball_x_y[0],
                self.score_l_r[0])
        else:
            raise ValueError(f'Invalid render mode: {self.render_mode}')

class RaylibClient:
    def __init__(self, width, height, paddle_width, paddle_height, ball_width, ball_height):
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.x_pad = 3 * self.paddle_width

        self.background_color = (0, 0, 0, 255)
        self.paddle_left_color = (0, 255, 0, 255)
        self.paddle_right_color = (255, 0, 0, 255)
        self.ball_color = (255, 255, 255, 255)

        rl.InitWindow(width + 2 * self.x_pad, height, "PufferLib MyPong".encode())
        rl.SetTargetFPS(15)  # 60 / frame_skip

    def render(self, paddles_pos, ball_pos, scores):
        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        rl.BeginDrawing()
        rl.ClearBackground(self.background_color)

        paddle_left = (self.x_pad - self.paddle_width,
                       self.height - paddles_pos[0] - self.paddle_height,
                       self.paddle_width, self.paddle_height)
        paddle_right = (self.width + self.x_pad,
                        self.height - paddles_pos[1] - self.paddle_height,
                        self.paddle_width, self.paddle_height)
        ball = (self.x_pad + ball_pos[0],
                self.height - ball_pos[1] - self.ball_height,
                self.ball_width, self.ball_height)

        rl.DrawRectangle(*map(int, paddle_left), self.paddle_left_color)
        rl.DrawRectangle(*map(int, paddle_right), self.paddle_right_color)
        rl.DrawRectangle(*map(int, ball), self.ball_color)
        for i in range(11):
            rl.DrawRectangle(int(self.width / 2) + self.x_pad - 3, 2 * i * self.height // 21, 6, self.height // 21, (255, 255, 255, 255))
        rl.DrawFPS(10, 10)
        str_scores = [str(scores[0]).encode('UTF-8'), str(scores[1]).encode('UTF-8')]
        rl.DrawText(str_scores[0], int(self.width / 2 + self.x_pad - 50 - rl.MeasureText(str_scores[0], 30) / 2), 10, 30, (255, 255, 255, 255))
        rl.DrawText(str_scores[1], int(self.width / 2 + self.x_pad + 50 - rl.MeasureText(str_scores[1], 30) / 2), 10, 30, (255, 255, 255, 255))
        rl.EndDrawing()

        return render.cdata_to_numpy()

def test_render(timeout=100, atn_cache=1024):
    env = MyPong(num_envs=1, render_mode='human')
    env.reset()

    while True:
        env.step([0] * env.num_envs)
        env.render()

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
    # test_performance()
    test_render()
