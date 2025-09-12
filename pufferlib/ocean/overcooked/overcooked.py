'''Overcooked: A multi-agent cooking coordination environment.'''

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.overcooked import binding

class Overcooked(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, width=5, height=5, num_agents=2,
                 render_mode=None, log_interval=128, buf=None, seed=0,
                 max_steps=1000, grid_size=32, 
                 reward_dish_served=10.0, reward_step_penalty=-0.01):
        
        # Define observation space - One-hot encoded grid
        # Channels: 8 terrain + 5 items + (2*num_agents) agent + 4 cooking = total channels
        n_terrain_channels = 8
        n_item_channels = 5  
        n_agent_channels = num_agents * 2
        n_cooking_channels = 4
        n_channels = n_terrain_channels + n_item_channels + n_agent_channels + n_cooking_channels
        
        grid_obs_size = width * height * n_channels  # (width, height, channels) flattened
        global_state_size = 2  # Time progress + dishes served
        observation_size = grid_obs_size + global_state_size
        
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