import numpy as np
import gymnasium
from .cy_bandit_cy import CBanditCy

class BanditCyEnv(gymnasium.Env):
    def __init__(self, num_actions=4, reward_scale=1, reward_noise=0, hard_fixed_seed=42):
        super().__init__()

        self.num_actions = num_actions
        self.reward_scale = reward_scale
        self.reward_noise = reward_noise
        self.hard_fixed_seed = hard_fixed_seed

        self.rewards = np.zeros((1, 1), dtype=np.float32)
        self.actions = np.zeros((1, 1), dtype=np.int32)

        self.c_env = CBanditCy(num_actions, 
                               reward_scale, 
                               reward_noise, 
                               hard_fixed_seed, 
                               self.rewards, 
                               self.actions)

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(num_actions)

    def reset(self, seed=None):
        self.c_env.reset()
        return np.ones(1, dtype=np.float32), {}

    def step(self, action):
        self.actions[0, 0] = action
        self.c_env.step()

        solution_idx = self.c_env.get_solution_idx()

        return np.ones(1, dtype=np.float32), self.rewards[0, 0], True, False, {'score': action == solution_idx}

    def render(self):
        pass

def test_performance(num_actions=4, timeout=10, atn_cache=1024):
    import time
    env = BanditCyEnv(num_actions=num_actions)

    env.reset()

    tick = 0
    actions = np.random.randint(0, num_actions, (atn_cache, 1))

    start = time.time()

    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn[0])
        tick += 1

    elapsed_time = time.time() - start
    sps = tick / elapsed_time
    print(f"SPS: {sps:.2f}")

if __name__ == '__main__':
    test_performance()
