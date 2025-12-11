import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.dinosaur import binding

class Dinosaur(pufferlib.PufferEnv):
    def __init__(self, num_envs=1024, width=800, height=800,
            speed_init=4, speed_max=8, obstacle_spawn_rate_init=120, obstacle_spawn_rate_min=70,
            rate_increment_rate=200, max_obstacles=8,
            render_mode=None, log_interval=128, size=11, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0.0, high=1,
            shape=(max_obstacles + 4,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(2)

        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i:i+1],
                self.actions[i:i+1],
                self.rewards[i:i+1],
                self.terminals[i:i+1],
                self.truncations[i:i+1],
                seed, width=width, height=height, speed_init=speed_init, speed_max=speed_max,
                obstacle_spawn_rate_init=obstacle_spawn_rate_init, obstacle_spawn_rate_min=obstacle_spawn_rate_min,
                rate_increment_rate=rate_increment_rate, max_obstacles=max_obstacles
            )
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

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
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 512

    env = Dinosaur(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.nvec, size=(CACHE, 1))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Dinosaur SPS:', int(steps / (time.time() - start)))
