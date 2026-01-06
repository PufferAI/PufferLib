'''2048 Gymnasium-compatible environment using the C backend.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.g2048 import binding

class G2048(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, reward_scaler=1.0,
                 can_go_over_65536=False, endgame_env_prob=0.0, scaffolding_ratio=0.0,
                 use_heuristic_rewards=False, snake_reward_weight=0.0, use_sparse_reward=False,
                 render_mode=None, log_interval=128, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=100, shape=(16*18 + 1,), dtype=np.uint8
        )
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        self.can_go_over_65536 = can_go_over_65536
        self.reward_scaler = reward_scaler
        self.endgame_env_prob = endgame_env_prob
        self.scaffolding_ratio = scaffolding_ratio
        self.use_heuristic_rewards = use_heuristic_rewards
        self.snake_reward_weight = snake_reward_weight
        self.use_sparse_reward = use_sparse_reward

        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            can_go_over_65536 = self.can_go_over_65536,
            reward_scaler = self.reward_scaler,
            endgame_env_prob = self.endgame_env_prob,
            scaffolding_ratio = self.scaffolding_ratio,
            use_heuristic_rewards = self.use_heuristic_rewards,
            snake_reward_weight = self.snake_reward_weight,
            use_sparse_reward = self.use_sparse_reward
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (
            self.observations, self.rewards,
            self.terminals, self.truncations, info
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 128

    env = G2048(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 4, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('2048 SPS:', int(steps / (time.time() - start)))

    env.close()
