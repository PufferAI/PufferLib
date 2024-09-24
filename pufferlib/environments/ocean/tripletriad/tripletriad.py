import numpy as np
import gymnasium

import pufferlib
from pufferlib.environments.ocean.tripletriad.cy_tripletriad import CyTripleTriad

class MyTripleTriad(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1,
             width=990, height=1000, piece_width=192, piece_height=224, game_over=0, num_cards=0):
        super().__init__()

        # env
        self.num_envs = num_envs
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval

        # sim hparams (px, px/tick)
        self.width = width
        self.height = height
        self.piece_width = piece_width
        self.piece_height = piece_height
        self.game_over = game_over
        self.num_cards = num_cards

        # spaces
        self.num_obs = 114
        self.num_act = 15
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
        self.reward_sum = 0
        self.num_finished_games = 0
        self.score_sum = 0
        self.misc_logging = np.zeros((self.num_envs, 2,), dtype=np.uint32)


    def reset(self, seed=None):
        self.tick = 0
        self.c_envs = []

        for i in range(self.num_envs):
            # TODO: since single agent, could we just pass values by reference instead of (1,) array?
            self.c_envs.append(CyTripleTriad(self.actions[i:i+1],
                self.buf.observations[i], self.buf.rewards[i:i+1], self.buf.terminals[i:i+1], self.misc_logging[i],
                self.width, self.height, self.piece_width, self.piece_height, self.game_over, self.num_cards))
            self.c_envs[i].reset()

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions

        for i in range(self.num_envs):
            self.c_envs[i].step()

        # TODO: hacky way to convert uint8 to bool
        self.buf.terminals[:] = self.terminals_uint8.astype(bool)
        self.tick += 1
        info = {}
        self.reward_sum += self.buf.rewards.mean()
        finished_rounds_mask = self.misc_logging[:,0] == 1
        self.num_finished_games += np.sum(finished_rounds_mask)
        self.score_sum += self.misc_logging[finished_rounds_mask, 1].sum()
        if self.tick % self.report_interval == 0:
            info.update({
                'reward': self.reward_sum / self.report_interval,
                'score': self.score_sum / self.num_finished_games,
                'num_games': self.num_finished_games,
            })

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        self.c_envs[0].render()

def test_performance(timeout=10, atn_cache=1024):
    env = MyTripleTriad(num_envs=1000)
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