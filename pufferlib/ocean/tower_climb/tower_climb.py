import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.tower_climb.cy_tower_climb import CyTowerClimb


class TowerClimb(pufferlib.PufferEnv):
    def __init__(self, num_envs=4096, render_mode=None, report_interval=1,
            num_maps=100, reward_climb_row = .25, reward_fall_row = 0, reward_illegal_move = -0.01,
            reward_move_block = 0.2, buf = None):

        # env
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        
        self.num_obs = 228
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.num_obs,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(6)

        super().__init__(buf=buf)   
        self.c_envs = CyTowerClimb(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, num_maps, reward_climb_row, reward_fall_row,
            reward_illegal_move, reward_move_block)

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
