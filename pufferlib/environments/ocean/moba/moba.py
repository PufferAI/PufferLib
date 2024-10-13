from pdb import set_trace as T
import numpy as np
import os

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.environments.ocean.moba.cy_moba import CyMOBA
from pufferlib.environments.ocean.moba.cy_moba import entity_dtype, reward_dtype

MAP_OBS_N = 11*11*4
PLAYER_OBS_N = 26

class PufferMoba(pufferlib.PufferEnv):
    def __init__(self, num_envs=4, vision_range=5, agent_speed=1.0,
            discretize=True, reward_death=-1.0, reward_xp=0.006,
            reward_distance=0.05, reward_tower=3.0,
            report_interval=32, render_mode='human', buf=None):

        self.report_interval = report_interval
        self.render_mode = render_mode
        self.num_agents = 10*num_envs

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(MAP_OBS_N + PLAYER_OBS_N,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 7, 3, 2, 2, 2])

        super().__init__(buf=buf)
        self.c_envs = CyMOBA(self.observations, self.actions, self.rewards,
            self.terminals, num_envs, vision_range, agent_speed, True,
            reward_death, reward_xp, reward_distance, reward_tower)

    def reset(self, seed=0):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.actions[:, 0] = 100*(self.actions[:, 0] - 3)
        self.actions[:, 1] = 100*(self.actions[:, 1] - 3)
        self.c_envs.step()

        infos = []
        self.tick += 1
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            if log['episode_length'] > 0:
                infos.append(dict(pufferlib.utils.unroll_nested_dict(log)))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, infos)

    def render(self):
        for frame in range(12):
            self.c_envs.render(frame)

    def close(self):
        self.c_envs.close()


def test_performance(timeout=20, atn_cache=1024, num_envs=400):
    tick = 0

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', 10*num_envs*tick / (time.time() - start))

if __name__ == '__main__':
    # Run with c profile
    from cProfile import run
    num_envs = 400
    env = PufferMoba(num_envs=num_envs, report_interval=10000000)
    env.reset()
    actions = np.random.randint(0, env.single_action_space.nvec, (1024, 10*num_envs, 6))
    test_performance(20, 1024, num_envs)
    exit(0)

    run('test_performance(20)', 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)

    #test_performance(10)
