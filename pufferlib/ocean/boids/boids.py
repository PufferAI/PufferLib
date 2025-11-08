'''
High-perf Boids
Inspired by https://people.ece.cornell.edu/land/courses/ece4760/labs/s2021/Boids/Boids.html
'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.boids import binding

class Boids(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        buf=None,
        render_mode=None,
        seed=0,
        report_interval=1,
        num_boids=1,
        margin_turn_factor=1.0,
        centering_factor=0.0,
        avoid_factor=0.0,
        matching_factor=0.0,
        max_num_threads=0
    ):
        ACTION_SPACE_SIZE = 2
        self.num_agents = num_envs * num_boids
        self.num_boids = num_boids

        self.single_observation_space = gymnasium.spaces.Box(
            -1000.0, 1000.0, shape=(num_boids*4,), dtype=np.float32
        )
        
        #self.single_action_space = gymnasium.spaces.Box(
        #    -np.inf, np.inf, shape=(ACTION_SPACE_SIZE,), dtype=np.float32
        #)
        self.single_action_space = gymnasium.spaces.MultiDiscrete([5, 5])

        self.render_mode = render_mode
        self.report_interval = report_interval

        super().__init__(buf, binding, max_num_threads)
        self.actions = self.actions.astype(np.float32)

        # Create C binding with flattened action buffer
        # We need to manually create a flattened action buffer to pass to C
        #self.flat_actions = np.zeros((self.num_agents * ACTION_SPACE_SIZE), dtype=np.float32)
        
        c_envs = []
        for env_num in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[env_num*num_boids:(env_num+1)*num_boids],
                #self.flat_actions[env_num*num_boids*ACTION_SPACE_SIZE:(env_num+1)*num_boids*ACTION_SPACE_SIZE],
                self.actions[env_num*num_boids:(env_num+1)*num_boids],
                self.rewards[env_num*num_boids:(env_num+1)*num_boids],
                self.terminals[env_num*num_boids:(env_num+1)*num_boids],
                self.truncations[env_num*num_boids:(env_num+1)*num_boids],
                seed,
                num_boids=num_boids,
                report_interval=self.report_interval,
                margin_turn_factor=margin_turn_factor,
                centering_factor=centering_factor,
                avoid_factor=avoid_factor,
                matching_factor=matching_factor,
            ))
        
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        # Clip actions to valid range
        clipped_actions = (actions.astype(np.float32) - 2.0) / 4.0
        #clipped_actions = np.clip(actions, -1.0, 1.0)
        
        # Copy the clipped actions to our flat actions buffer for C binding
        # Flatten from [num_agents, num_boids, 2] to a 1D array for C
        # TODO: Check if I even need this? its not like I'm using the actions anywhere else
        #self.flat_actions[:] = clipped_actions.reshape(-1)
        
        # Save the original actions for the experience buffer
        # TODO: Same thing with this
        self.actions[:] = clipped_actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        # print(f"OBSERVATIONS: {self.observations}")
        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = Boids(num_envs=1000)
    env.reset()
    tick = 0

    # Generate random actions with proper shape: [cache_size, num_agents, action_dim]
    actions = np.random.uniform(-3.0, 3.0, (atn_cache, env.num_agents, 2))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: {env.num_agents * tick / (time.time() - start)}')


if __name__ == '__main__':
    test_performance()
