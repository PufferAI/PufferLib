import gymnasium 
import numpy as np 

import pufferlib 
from pufferlib.ocean.common_pool_resource.cy_cpr import CyEnv

class PyCPR(pufferlib.PufferEnv):
    def __init__(self, 
                num_envs=1,
                widths=[32],
                heights=[32], 
                num_agents=[8],  
                vision=3, 
                reward_food=1.0, 
                interactive_food_reward=5.0,
                reward_move=-0.01,
                food_base_spawn_rate=2e-3,
                report_interval=1,
                render_mode=None, 
                buf=None,
                seed=0,
            ):
        widths = num_envs*widths
        heights = num_envs*heights 
        num_agents = num_envs*num_agents 

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=255, shape=((2*vision+1)*(2*vision+1),), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = sum(num_agents)

        self.tick = 0
        self.report_interval = report_interval

        super().__init__(buf)

        self.c_envs = CyEnv(
            self.observations, 
            self.actions, 
            self.rewards, 
            self.terminals,
            self.truncations,
            self.masks,
            widths,
            heights,
            num_agents,
            vision, 
            reward_food,
            interactive_food_reward,
            reward_move,
            food_base_spawn_rate
        )

    def reset(self, seed=None):
        self.tick = 0
        self.c_envs.reset()

        return self.observations, []
    
    def step(self, actions):
        self.actions[:] = actions 
        self.c_envs.step()
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            info.append(log)
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()

    def close(self):
        self.c_envs.close()

if __name__ == "__main__":
    env = PyCPR()
    env.reset()
    tick = 0
    timeout=30

    tot_agents = env.num_agents
    actions = np.random.randint(0,5,(1024,tot_agents))

    import time 
    start = time.time()
    while time.time() - start < timeout:
    # while tick < 500:
        atns = actions[tick % 1024]
        env.step(atns)
        # if -1 in env.rewards:
        #     breakpoint()
        # env.render()
        tick += 1

    print(f'SPS: {int(tot_agents * tick / (time.time() - start)):_}')

    env.close()




