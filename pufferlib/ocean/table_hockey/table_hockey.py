import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.table_hockey import binding

class TableHockey(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode='headless', action_mode='continuous',
                 dt=1.0/60.0, max_paddle_speed=3.0, 
                 puck_hit_reward=0.01, goal_reward=1.0,
                 report_interval=1, buf=None, seed=0):
        
        self.render_mode_str = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.action_mode = action_mode

        render_mode_map = {
            'headless': 0,
            'human': 1,
            'vr': 2
        }
        self.render_mode = render_mode_map.get(render_mode, 0)
        action_mode_map = {
            'continuous': 0,
            'discrete': 1
        }
        self.action_mode_val = action_mode_map.get(action_mode, 0)
        self.num_obs = 11
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_obs,), dtype=np.float32
        )
        
        if action_mode == 'continuous':
            self.single_action_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )
        else:
            self.single_action_space = gymnasium.spaces.Discrete(9)
        
        super().__init__(buf)
        self.actions = np.zeros((num_envs, 2), dtype=np.float32)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            render_mode=self.render_mode,
            action_mode=self.action_mode_val,
            dt=dt,
            max_paddle_speed=max_paddle_speed,
            puck_hit_reward=puck_hit_reward,
            goal_reward=goal_reward,
        )
    
    def reset(self, seed=None):
        """Reset state"""
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []
    
    def step(self, actions):
        """Step environment"""
        if self.action_mode == 'continuous':
            self.actions[:] = np.clip(actions, -1.0, 1.0)
        else:
            self.actions[:, 0] = np.clip(actions.astype(np.float32), 0, 8)
            self.actions[:, 1] = 0

        self.tick += 1
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))
        
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info
        )
    
    def render(self):
        """Render the environment (if not headless)"""
        if self.render_mode != 0:
            binding.vec_render(self.c_envs, 0)
    
    def close(self):
        """Clean up environment resources"""
        binding.vec_close(self.c_envs)