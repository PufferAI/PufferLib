'''Overcooked: A single-agent cooking coordination environment.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.overcooked import binding

class Overcooked(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=10, height=10, 
                 render_mode=None, log_interval=128, buf=None, seed=0,
                 max_steps=1000, grid_size=32, 
                 reward_dish_served=10.0, reward_step_penalty=-0.01):
        
        # Define observation space
        # For now: 7x7 grid view + agent state (position, held item)
        grid_obs_size = 7 * 7 * 2  # 7x7 window, 2 channels (tiles, items)
        agent_state_size = 4  # x, y, held_item, facing_direction
        observation_size = grid_obs_size + agent_state_size
        
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1,
            shape=(observation_size,), 
            dtype=np.float32
        )
        
        # Action space: 6 discrete actions (noop, up, down, left, right, interact)
        self.single_action_space = gymnasium.spaces.Discrete(6)
        
        self.render_mode = render_mode
        self.num_agents = num_envs  # Single agent per env
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
                seed + i,
                width=width,
                height=height,
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
    # Test with single environment
    env = Overcooked(num_envs=1)
    env.reset()
    steps = 0
    
    import time
    start = time.time()
    
    # Run for 10 seconds with random actions
    while time.time() - start < 10:
        # Random action from action space
        action = np.random.randint(0, 6, size=(1,))
        obs, rewards, dones, truncs, info = env.step(action)
        
        if env.render_mode:
            env.render()
        
        steps += 1
        
        # Reset if done
        if dones[0]:
            env.reset()
    
    print('Overcooked SPS:', int(steps / (time.time() - start)))