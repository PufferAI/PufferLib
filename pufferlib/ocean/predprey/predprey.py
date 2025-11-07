import gymnasium 
import numpy as np 

import pufferlib 
from pufferlib.ocean.predprey import binding

class PredPrey(pufferlib.PufferEnv):
    def __init__(self, 
                num_envs=1,
                width=32,
                height=32, 
                num_agents=8,  
                vision=3, 
                food_base_spawn_rate=1e-3,
                reward_death_scale = 1.0,
                reward_eat = 0,
                reward_collect = 0,
                timestep_reward = 0,
                reward_steal = 0,
                hp_reward_scale = 0,
                held_food_reward_scale = 0,                report_interval=1,
                render_mode=None, 
                buf=None,
                seed=0,
            ):
        obs_shape = ((2*vision+1)*(2*vision+1)*5)+1
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(obs_shape,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(7)
        self.render_mode = render_mode
        self.num_agents = num_agents * num_envs

        self.tick = 0
        self.report_interval = report_interval

        super().__init__(buf)
        c_envs = []
        for i in range(num_envs):
            n = num_agents
            env_id = binding.env_init(
                self.observations[i*n:(i+1)*n],
                self.actions[i*n:(i+1)*n],
                self.rewards[i*n:(i+1)*n],
                self.terminals[i*n:(i+1)*n],
                self.truncations[i*n:(i+1)*n],
                i + seed * num_envs,
                width=width,
                height=height,
                num_agents=num_agents,
                vision=vision,
                reward_death_scale = reward_death_scale,
                reward_eat = reward_eat,
                reward_collect = reward_collect,
                timestep_reward = timestep_reward,
                reward_steal = reward_steal,
                hp_reward_scale = hp_reward_scale,
                held_food_reward_scale = held_food_reward_scale,
                food_base_spawn_rate=food_base_spawn_rate,
            )
            c_envs.append(env_id)

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=0):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []
    
    def step(self, actions):
        self.actions[:] = actions 
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def pret(obs, i):
    for j in range(7):
        print(obs[i,j*7*3:(j+1)*7*3])
        print("************************")

if __name__ == "__main__":
    print("Testing PredatorPrey CEnv")
    
    env = PredPrey()
    o, _ = env.reset()
    tick = 0
    timeout=30

    tot_agents = env.num_agents
    actions = np.random.randint(0,7,(1024,tot_agents))

    env.render()
    import time 
    start = time.time()
    # while tick < 1000:
    while time.time() - start < timeout:
        atns = actions[tick % 1024]
        o, r, t, trun, info = env.step(atns)
        env.render()
        tick += 1

    print(f'SPS: {int(tot_agents * tick / (time.time() - start)):_}')

    env.close()




