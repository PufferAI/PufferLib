
import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.four_rooms import binding

class FourRooms(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=19, buf=None, seed=0):
        # MinGrid-compatible observation space: 7x7x3 (OBJECT_IDX, COLOR_IDX, STATE)
        # Flattened to 147 elements for PufferLib compatibility  
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=10,
            shape=(7*7*3,), dtype=np.uint8)
        # MinGrid-compatible action space: 7 actions (only first 3 used in FourRooms)
        # 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done
        self.single_action_space = gymnasium.spaces.Discrete(7)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval

        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size)
 
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

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 4096

    env = FourRooms(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 7, (CACHE, N))  # 7 actions: left, right, forward, pickup, drop, toggle, done

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('FourRooms SPS:', int(steps / (time.time() - start)))
