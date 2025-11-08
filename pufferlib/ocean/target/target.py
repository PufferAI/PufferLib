'''A simple sample environment. Use this as a template for your own envs.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.target import binding

class Target(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=1080, height=720, num_agents=8,
            num_goals=4, render_mode=None, log_interval=128, size=11, buf=None, seed=0,
            max_num_threads=0):
        # Observation space: Each agent observes how close they are to the goals, and the other agents (including self).
        # NOTE: Distance to self for each agent is (0, 0).
        # 4 additional features (heading, reward, agent (self) speed, agent (self) heading, etc.)
        # All x,y coordinates are normalized to [0, 1] range.
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(2*(num_agents+num_goals) + 4,), dtype=np.float32)
        # See https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete
        # Heading: 9 discrete actions, Speed: 4 discrete speeds.
        self.single_action_space = gymnasium.spaces.MultiDiscrete([9, 5])

        self.render_mode = render_mode
        self.num_agents = num_envs*num_agents
        self.log_interval = log_interval

        super().__init__(buf, binding, max_num_threads)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                seed, width=width, height=height,
                num_agents=num_agents, num_goals=num_goals)
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

    env = Target(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(env.single_action_space.nvec, size=(CACHE, 2))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Target SPS:', int(steps / (time.time() - start)))
