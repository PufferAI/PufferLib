import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.boxoban import binding

class Boxoban(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=10, buf=None, seed=0, difficulty=0, max_steps = 500,int_r_coeff = 0.1, target_loss_pen_coeff = 0.5):
        self.shape = size*size*4 #agents walls boxes targets OHE
        self.difficulty = difficulty
        self._difficulty_stat_key = f"difficulty ({self.difficulty})"
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.shape,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.max_steps = max_steps
        self.int_r_coeff = int_r_coeff
        self.target_loss_pen_coeff = target_loss_pen_coeff

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size, max_steps = self.max_steps, int_r_coeff = self.int_r_coeff, target_loss_pen_coeff = self.target_loss_pen_coeff, difficulty=self.difficulty)
 
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
            log_dict = binding.vec_log(self.c_envs)
            log_dict[self._difficulty_stat_key] = 1.0
            info.append(log_dict)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 1

    env = Boxoban(num_envs=N)
    env.reset()
    env.render()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('Boxoban SPS:', int(steps / (time.time() - start)))
