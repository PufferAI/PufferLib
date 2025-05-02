'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.go import binding

class Go(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=1,
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
            buf = None, seed=0):

        # env
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.tick = 0
        self.num_obs = (grid_size) * (grid_size)*2 + 2
        self.num_act = (grid_size) * (grid_size) + 1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(self.num_act)

        super().__init__(buf=buf)
        height = 64*(grid_size+1)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, width=width, height=height, grid_size=grid_size,
            board_width=board_width, board_height=board_height, grid_square_size=grid_square_size,
            moves_made=moves_made, komi=komi, score=score, last_capture_position=last_capture_position,
            reward_move_pass=reward_move_pass, reward_move_invalid=reward_move_invalid,
            reward_move_valid=reward_move_valid, reward_player_capture=reward_player_capture,
            reward_opponent_capture=reward_opponent_capture)

    def reset(self, seed=None):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))
            
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)
        
    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    num_envs=1000
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
