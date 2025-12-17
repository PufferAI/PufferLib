'''Ant Colony Simulation Environment - Simplified following Target pattern'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.ants import binding

class AntsEnv(pufferlib.PufferEnv):
    """
    Ant Colony Simulation Environment

    Two colonies compete to collect food from the environment.
    Simplified architecture following the Target environment pattern.

    Observations (8 per ant):
        - colony_dx, colony_dy: Direction to home colony (normalized)
        - food_dx, food_dy: Direction to nearest VISIBLE food (normalized, with vision constraints)
        - pheromone_dx, pheromone_dy: Direction to nearest VISIBLE pheromone from own colony
        - has_food: Binary flag (0 or 1)
        - heading: Ant's current direction (normalized)

    Vision System:
        - Ants have limited vision range (100 pixels) - very short range
        - Vision cone of 30 degrees (π/6) - narrow focused beam
        - Can only see food and pheromones within their vision cone

    Pheromone System:
        - Ants automatically drop pheromones every 5 steps while carrying food
        - Pheromones evaporate over time (rate: 0.001 per step)
        - Each colony's pheromones are distinct
        - Ants only observe pheromones from their own colony

    Actions (Discrete 4):
        0: Turn left
        1: Turn right
        2: Move forward
        3: No-op
    """

    def __init__(
            self,
            num_envs=1,
            width=1280,
            height=720,
            num_ants=32,
            reward_food_pickup=0.1,
            reward_delivery=10.0,
            render_mode=None,
            log_interval=128,
            buf=None,
            seed=0):

        # Observation space: 8 values per ant (added pheromone observations)
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        # Discrete action space: turn left, turn right, move forward, noop
        self.single_action_space = gymnasium.spaces.Discrete(4)

        self.render_mode = render_mode
        self.num_agents = num_envs * num_ants
        self.log_interval = log_interval

        super().__init__(buf)

        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_ants:(i+1)*num_ants],
                self.actions[i*num_ants:(i+1)*num_ants],
                self.rewards[i*num_ants:(i+1)*num_ants],
                self.terminals[i*num_ants:(i+1)*num_ants],
                self.truncations[i*num_ants:(i+1)*num_ants],
                seed + i,  # Unique seed per env
                width=width,
                height=height,
                num_ants=num_ants,
                reward_food_pickup=reward_food_pickup,
                reward_delivery=reward_delivery
            )
            c_envs.append(c_env)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        """Reset all environments"""
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        """Execute one step for all agents"""
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
        """Render the first environment"""
        binding.vec_render(self.c_envs, 0)

    def close(self):
        """Clean up resources"""
        binding.vec_close(self.c_envs)


if __name__ == '__main__':
    # Performance test following target pattern
    N = 512

    env = AntsEnv(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(4, size=(CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += env.num_agents
        i += 1

    print('Ants SPS:', int(steps / (time.time() - start)))
