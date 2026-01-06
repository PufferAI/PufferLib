'''A minimal matsci for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.matsci import binding

class Matsci(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, num_atoms=2, render_mode=None, log_interval=128, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(3,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(3,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.num_agents = num_envs*num_atoms

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i*num_atoms:(i+1)*num_atoms],
                self.actions[i*num_atoms:(i+1)*num_atoms],
                self.rewards[i*num_atoms:(i+1)*num_atoms],
                self.terminals[i*num_atoms:(i+1)*num_atoms],
                self.truncations[i*num_atoms:(i+1)*num_atoms],
                i,
                num_agents=num_atoms,
            ))

        self.c_envs = binding.vectorize(*c_envs)
 
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
    env = Matsci(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1

    print('Squared SPS:', int(env.num_agents*steps / (time.time() - start)))
