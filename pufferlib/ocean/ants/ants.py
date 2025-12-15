import numpy as np
import gymnasium

import pufferlib
from pufferlib import APIUsageError
from pufferlib.ocean.ants import binding

class AntsEnv(pufferlib.PufferEnv):
    """
    Ant Colony Simulation Environment
    
    Each ant receives observations about its surroundings and can:
    - Move forward (always happens)
    - Turn left/right
    - Drop pheromone trails
    
    Two colonies compete to collect food from the environment.
    Following multiagent architecture patterns from snake environment.
    """
    
    def __init__(
            self,
            num_envs=32,
            width=1280,
            height=720,
            num_ants=32,
            reward_food=0.1,
            reward_delivery=5.0,
            reward_death=0.0,
            reward_demo_match=0.001,
            reward_demo_mismatch=-0.001,
            report_interval=1,
            render_mode=None,
            buf=None,
            seed=0):
        
        if num_envs is not None:
            num_ants = num_envs * [num_ants]
            width = num_envs * [width]
            height = num_envs * [height]
        
        if not (len(num_ants) == len(width) == len(height)):
            raise APIUsageError('num_ants, width, height must be lists of equal length')
        
        for w, h in zip(width, height):
            if w < 100 or h < 100:
                raise APIUsageError('width and height must be at least 100')
        
        self.report_interval = report_interval
        self.num_agents = sum(num_ants)
        self.render_mode = render_mode
        self.tick = 0
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Calculate cell size for rendering
        self.cell_size = int(np.ceil(1280 / max(max(width), max(height))))

        super().__init__(buf)

        c_envs = []
        offset = 0
        for i in range(num_envs):
            na = num_ants[i]
            obs_slice = self.observations[offset:offset+na*8]  # Multiply by obs_size
            act_slice = self.actions[offset:offset+na]
            rew_slice = self.rewards[offset:offset+na]
            term_slice = self.terminals[offset:offset+na]
            trunc_slice = self.truncations[offset:offset+na]

            # Seed each env uniquely: i + seed * num_envs
            env_seed = i + seed * num_envs
            env_id = binding.env_init(
                obs_slice,
                act_slice,
                rew_slice,
                term_slice,
                trunc_slice,
                env_seed,
                width=width[i],
                height=height[i],
                num_ants=na,
                reward_food=reward_food,
                reward_delivery=reward_delivery,
                reward_death=reward_death,
                reward_demo_match=reward_demo_match,
                reward_demo_mismatch=reward_demo_mismatch,
                cell_size=self.cell_size
            )
            c_envs.append(env_id)
            offset += na * 8  # Multiply by obs_size
        
        # VECTORIZE ENVIRONMENTS - FOLLOWING SNAKE PATTERN
        self.c_envs = binding.vectorize(*c_envs)
    
    def reset(self, seed=None):
        """Reset all environments"""
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []
    
    def step(self, actions):
        """Execute one step for all agents"""
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                # Add computed metrics
                info.append(log_data)
        
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)
    
    def render(self):
        """Render the first environment"""
        binding.vec_render(self.c_envs, 0)
    
    def close(self):
        """Clean up resources"""
        binding.vec_close(self.c_envs)


def test_performance(timeout=10, atn_cache=1024):
    """Performance test following snake pattern"""
    env = AntsEnv(num_envs=64, num_ants=50)
    env.reset()
    tick = 0
    
    total_ants = env.num_agents
    actions = np.random.randint(0, 4, (atn_cache, total_ants))
    
    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        obs, rewards, dones, truncs, info = env.step(atns)
        
        # Print info when available
        if info:
            for log_data in info:
                if 'n' in log_data and log_data['n'] > 0:
                    print(f"Tick {tick}: Episodes: {log_data['n']:.0f}, "
                          f"Avg score: {log_data.get('score', 0) / log_data['n']:.2f}, "
                          f"Avg return: {log_data.get('episode_return', 0) / log_data['n']:.2f}")
        
        tick += 1
    
    elapsed = time.time() - start
    sps = total_ants * tick / elapsed
    print(f'Ant SPS: {sps:.0f} ({tick} environment steps)')
    env.close()


if __name__ == '__main__':
    test_performance()