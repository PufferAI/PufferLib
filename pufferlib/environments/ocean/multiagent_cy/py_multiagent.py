import numpy as np
import gymnasium
import pettingzoo
from .cy_multiagent_cy import CMultiagentCy

# Suppress clean_pufferl printouts
import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], str) and 'Reward:' not in args[0]:
        original_print(*args, **kwargs)
builtins.print = custom_print   

class MultiagentCyEnv(pettingzoo.ParallelEnv):
    def __init__(self, num_agents=2):
        super().__init__()

        self.observation1 = np.zeros(1, dtype=np.float32)
        self.observation2 = np.ones(1, dtype=np.float32)
        self.actions = np.zeros(2, dtype=np.int32)
        self.rewards = np.zeros(2, dtype=np.float32)
        self.view = np.zeros((2, 5), dtype=np.float32)

        self.c_env = CMultiagentCy(
            num_agents,
            self.observation1, 
            self.observation2, 
            self.actions, 
            self.rewards, 
            self.view
        )

        self.possible_agents = [1, 2]
        self.agents = [1, 2]
        self.render_mode = 'ansi'
        self.tick_calc = 0
        self.reward_total = 0

    def observation_space(self, agent):
        return gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def action_space(self, agent):
        return gymnasium.spaces.Discrete(2)

    def reset(self, seed=None):
        self.c_env.reset()

        return {1: self.observation1, 2: self.observation2}, {}

    def step(self, actions):
        self.actions[0] = actions[1]
        self.actions[1] = actions[2]
        self.c_env.step()

        rewards = {1: self.rewards[0], 2: self.rewards[1]}

        infos = {
            1: {'score': self.rewards[0]},
            2: {'score': self.rewards[1]}
        }

        return {1: self.observation1, 2: self.observation2}, rewards, {1: True, 2: True}, {1: False, 2: False}, infos        

    def render(self):
        def _render(val):
            if val == 1:
                return f'\033[94m██\033[0m'
            return f'\033[90m██\033[0m'

        chars = []

        for row in self.view:
            for val in row:
                chars.append(_render(val))
            chars.append('\n')

        grid = ''.join(chars)

        self.tick_calc += 1

        agent1_reward, agent2_reward = self.rewards[0], self.rewards[1]

        if not hasattr(self, 'agent1_total_reward'):
            self.agent1_total_reward = 0.0
            self.agent2_total_reward = 0.0

        self.agent1_total_reward += agent1_reward
        self.agent2_total_reward += agent2_reward

        average_agent1_score = self.agent1_total_reward / self.tick_calc
        average_agent2_score = self.agent2_total_reward / self.tick_calc

        tick_info = (f"Tick: {self.tick_calc}\n"
                    f"Agent 1 Average Score: {average_agent1_score:.4f}\n"
                    f"Agent 2 Average Score: {average_agent2_score:.4f}\n")

        return grid + tick_info

    def close(self):
        pass


def test_performance(num_agents=2, atn_cache=1024, timeout=10):
    import time
    import numpy as np

    env = MultiagentCyEnv(num_agents=num_agents)
    env.reset()
    tick = 0
    actions_cache = {agent: np.random.randint(0, 2, atn_cache) for agent in env.possible_agents}
    start = time.time()
    while time.time() - start < timeout:
        actions = {agent: actions_cache[agent][tick % atn_cache] for agent in env.possible_agents}
        env.step(actions)
        tick += 1
    elapsed_time = time.time() - start
    sps = tick / elapsed_time
    print(f"SPS: {sps:.2f}")

if __name__ == '__main__':
    test_performance()
