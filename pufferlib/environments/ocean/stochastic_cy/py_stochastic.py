import numpy as np
import gymnasium
import random
from .cy_stochastic_cy import CStochasticCy

class StochasticCyEnv(gymnasium.Env):
    '''Pufferlib Stochastic environment in Cython

    The optimal policy is to play action 0 < p % of the time and action 1 < (1 - p) %.
    This tests whether the algorithm can learn a nontrivial stochastic policy.

    Observation space: Box(0, 1, (1,))
    Action space: Discrete(2)
    '''
    def __init__(self, p=0.75, horizon=1000):
        super().__init__()
        self.p = p
        self.horizon = horizon
        self.c_env = CStochasticCy(p, horizon)
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = gymnasium.spaces.Discrete(2)
        self.render_mode = 'ansi'
        self.tick_calc = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.c_env.reset()
        
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        reward, done, info = self.c_env.step(action)
        self.tick_calc += 1

        return np.zeros(1, dtype=np.float32), reward, done, False, info

    def render(self):
        def _render(val):
            if val == 1:
                c = 94
            elif val == 0:
                c = 91
            else:
                c = 90
            return f'\033[{c}m██\033[0m'
        chars = []
        if self.tick_calc == 0:
            solution = 0
        else:
            solution = 0 if self.c_env.get_count() / self.tick_calc < self.p else 1
        chars.append(_render(solution))
        chars.append(' Solution\n')

        action = self.c_env.get_action()
        chars.append(_render(action))
        chars.append(' Prediction\n')

        return ''.join(chars)

    def close(self):
        pass

def test_performance(p=0.7, horizon=100, atn_cache=1024, timeout=10):
    import time

    env = StochasticCyEnv(p=p, horizon=horizon)
    env.reset()
    tick = 0
    actions_cache = np.random.randint(0, 2, atn_cache)
    start = time.time()
    while time.time() - start < timeout:
        action = actions_cache[tick % atn_cache]
        env.step(action)
        tick += 1
    elapsed_time = time.time() - start
    sps = tick / elapsed_time
    print(f"SPS: {sps:.2f}")

if __name__ == '__main__':
    test_performance()