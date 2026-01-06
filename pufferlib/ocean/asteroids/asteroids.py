import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.asteroids import binding

class Asteroids(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, buf=None, seed=0, size=500, frameskip=4):
        obs_shape = 4 + 5 * 20  # player pos, player vel, [asteroid pos, asteroid vel, asteroid size] x num asteroids
        self.single_observation_space = gymnasium.spaces.Box(low=-5, high=5,
            shape=(obs_shape,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(4)  # forward, left, right, shoot
        self.render_mode = render_mode
        self.num_agents = num_envs

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size, frameskip=frameskip)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 4096
    env = Asteroids(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    sps = int(env.num_agents*steps / (time.time() - start))
    print(f'Asteroids SPS: {sps:,}')
