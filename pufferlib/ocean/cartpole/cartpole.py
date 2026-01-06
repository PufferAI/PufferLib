import numpy as np
import gymnasium
import pufferlib
from pufferlib.ocean.cartpole import binding

class Cartpole(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, cart_mass=1.0, pole_mass=0.1,
            pole_length=0.5, gravity=9.8, force_mag=10.0, dt=0.02,
            render_mode='human', report_interval=1, continuous=False,
            buf=None, seed=0):
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.report_interval = report_interval
        self.tick = 0
        self.continuous = continuous
        self.human_action = None

        self.num_obs = 4
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        if self.continuous:
            self.single_action_space = gymnasium.spaces.Box(
                low=-1.0, high=1.0, shape=(1,)
            )
            
        else:
            self.single_action_space = gymnasium.spaces.Discrete(2)

        super().__init__(buf)
        self.actions = np.zeros(num_envs, dtype=np.float32)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            cart_mass=cart_mass,
            pole_mass=pole_mass,
            pole_length=pole_length,
            gravity=gravity,
            force_mag=force_mag,
            dt=dt,
            continuous=int(self.continuous),
        )
   
    def reset(self, seed=None):
        self.tick = 0      
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []
   
    def step(self, actions):
        if self.continuous:
            self.actions[:] = np.clip(actions.flatten(), -1.0, 1.0)
        else:
            self.actions[:] = actions
        
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
        binding.vec_render(self.c_envs, 0)
   
    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=8192, continuous=True):
    """Benchmark environment performance."""
    num_envs = 4096
    env = Cartpole(num_envs=num_envs, continuous=continuous)
    env.reset()
    tick = 0

    if env.continuous:
        actions = np.random.uniform(-1, 1, (atn_cache, num_envs, 1)).astype(np.float32)
    else:
        actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_envs)).astype(np.int8)

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1
    sps = num_envs * tick / (time.time() - start)
    print(f'SPS: {sps:,}')

if __name__ == '__main__':
    test_performance()
    
