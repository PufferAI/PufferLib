'''Overcooked: A multi-agent cooking coordination environment.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.overcooked import binding

class Overcooked(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=5, height=5, num_agents=2,
                 render_mode=None, log_interval=128, buf=None, seed=0,
                 max_steps=400, grid_size=32, 
                 reward_dish_served=20.0, reward_step_penalty=0.0):
        
        # Define observation space - 83-dimensional vector per agent (was 77, added 4 wall + 2 soup ingredients)
        # Structure:
        # - Player features: 34 dims
        #   * Orientation (one-hot): 4
        #   * Held object (one-hot): 4
        #   * Proximity to objects (dx,dy): 12 (6 objects × 2)
        #   * Nearest soup ingredients: 2 (onions, tomatoes in plated soup or held)
        #   * Pot soup ingredients: 2 (onions, tomatoes in nearest pot)
        #   * Pot existence: 1
        #   * Pot state flags: 4
        #   * Cooking time: 1
        #   * Wall detection: 4 (up, down, left, right)
        # - Teammate features: 46 dims (28 mirrored + 18 simplified + 2 relative pos)
        # - Absolute position: 2 dims
        # - Reward: 1 dim
        # Total: 83 dimensions (including reward)

        observation_size = 83  # Including reward
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1,
            shape=(observation_size,), 
            dtype=np.float32
        )
        
        # Action space: 6 discrete actions (noop, up, down, left, right, interact)
        self.single_action_space = gymnasium.spaces.Discrete(6)
        
        self.render_mode = render_mode
        self.num_agents = num_envs * num_agents  # Multiple agents per env
        self.log_interval = log_interval
        
        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            c_env = binding.env_init(
                self.observations[i*num_agents:(i+1)*num_agents],
                self.actions[i*num_agents:(i+1)*num_agents],
                self.rewards[i*num_agents:(i+1)*num_agents],
                self.terminals[i*num_agents:(i+1)*num_agents],
                self.truncations[i*num_agents:(i+1)*num_agents],
                seed + i,
                width=width,
                height=height,
                num_agents=num_agents,
                max_steps=max_steps,
                grid_size=grid_size,
                observation_size=observation_size,
                reward_dish_served=reward_dish_served,
                reward_step_penalty=reward_step_penalty
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
    # Test with single environment, 2 agents
    num_agents = 2
    env = Overcooked(num_envs=1, num_agents=num_agents)
    env.reset()
    steps = 0
    
    import time
    start = time.time()
    
    # Run for 10 seconds with random actions
    while time.time() - start < 10:
        # Random action from action space for both agents
        actions = np.random.randint(0, 6, size=(num_agents,))
        obs, rewards, dones, truncs, info = env.step(actions)
        
        if env.render_mode:
            env.render()
        
        steps += num_agents  # Count steps for all agents
        
        # Reset if any agent is done
        if any(dones):
            env.reset()
    
    print('Overcooked SPS:', int(steps / (time.time() - start)))