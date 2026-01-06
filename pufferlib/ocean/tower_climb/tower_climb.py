import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.tower_climb import binding


class TowerClimb(pufferlib.PufferEnv):
    def __init__(self, num_envs=4096, render_mode=None, report_interval=1,
            num_maps=50, reward_climb_row = .25, reward_fall_row = 0, reward_illegal_move = -0.01,
            reward_move_block = 0.2, buf = None, seed=0):

        # env
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        
        self.num_obs = 228
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.num_obs,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(6)

        super().__init__(buf=buf)   
        c_envs = []
        self.c_state = binding.shared(num_maps=num_maps)
        self.c_envs = binding.vec_init(self.observations, self.actions,
            self.rewards, self.terminals, self.truncations, num_envs, seed,
            num_maps=num_maps, reward_climb_row=reward_climb_row,
            reward_fall_row=reward_fall_row, reward_illegal_move=reward_illegal_move,
            reward_move_block=reward_move_block, state=self.c_state)

    def reset(self, seed=None):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)
        
    def close(self):
        #binding.vec_close(self.c_envs)
        pass

def test_performance(timeout=10, atn_cache=1024):
    num_envs=1000;
    env = TowerClimb(num_envs=num_envs)
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
