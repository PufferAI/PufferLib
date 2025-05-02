'''High-perf Pong

Inspired from https://gist.github.com/Yttrmin/18ecc3d2d68b407b4be1
& https://jair.org/index.php/jair/article/view/10819/25823
& https://www.youtube.com/watch?v=PSQt5KGv7Vk
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.rware.cy_rware import CyRware

PLAYER_OBS_N = 27

class Rware(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
            width=1280, height=1024,
            num_agents=4,
            map_choice=1,
            num_requested_shelves=4,
            grid_square_size=64,
            human_agent_idx=0,
            reward_type=1,
            buf = None, seed=0):

        # env
        self.num_agents = num_envs*num_agents
        self.render_mode = render_mode
        self.report_interval = report_interval
        
        self.num_obs = 27
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(5)

        super().__init__(buf=buf)
        self.c_envs = CyRware(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, width, height, map_choice, num_agents, num_requested_shelves, grid_square_size, human_agent_idx)


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
    env = MyRware(num_envs=num_envs)
    env.reset()
    tick = 0

    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, 5*num_envs))

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
