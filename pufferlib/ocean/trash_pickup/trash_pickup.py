import numpy as np
from gymnasium import spaces

import pufferlib
from pufferlib.ocean.trash_pickup.cy_trash_pickup import CyTrashPickup


class TrashPickupEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, report_interval=1, buf=None, 
                 grid_size=10, num_agents=3, num_trash=15, num_bins=2, max_steps=300, agent_sight_range=5, seed=0):
        # Env Setup
        self.render_mode = render_mode
        self.report_interval = report_interval

        # Validate num_agents
        if not isinstance(num_agents, int) or num_agents <= 0:
            raise ValueError("num_agents must be an integer greater than 0.")
        self.num_agents = num_envs * num_agents
        self.num_agents_per_env = num_agents

        # Handle num_trash input
        if not isinstance(num_trash, int) or num_trash <= 0:
            raise ValueError("num_trash must be an int > 0")
        self.num_trash = num_trash

        # Handle num_bins input
        if not isinstance(num_bins, int) or num_bins <= 0:
            raise ValueError("num_bins must be an int > 0")
        self.num_bins = num_bins

        if not isinstance(max_steps, int) or max_steps < 10:
            raise ValueError("max_steps must be an int >= 10")
        self.max_steps = max_steps

        if not isinstance(agent_sight_range, int) or agent_sight_range < 2:
            raise ValueError("agent sight range must be an int >= 2")
        self.agent_sight_range = agent_sight_range

        # Calculate minimum required grid size
        min_grid_size = int((num_agents + self.num_trash + self.num_bins) ** 0.5) + 1
        if not isinstance(grid_size, int) or grid_size < min_grid_size:
            raise ValueError(
                f"grid_size must be an integer >= {min_grid_size}. "
                f"Received grid_size={grid_size}, with num_agents={num_agents}, num_trash={self.num_trash}, and num_bins={self.num_bins}."
            )
        self.grid_size = grid_size

        # Entity Attribute Based Obs-Space
        # num_obs_trash = num_trash * 3  # [presence, x pos, y pos] for each trash
        # num_obs_bin = num_bins * 2  # [x pos, y pos] for each bin
        # num_obs_agent = num_agents * 3  # [carrying trash, x pos, y pos] for each agent
        # self.num_obs = num_obs_trash + num_obs_bin + num_obs_agent;
        
        # 2D Local crop obs space
        self.num_obs = ((((agent_sight_range * 2 + 1) * (agent_sight_range * 2 + 1)) * 5));  # one-hot encoding for all cell types in local crop around agent (minus the cell the agent is currently in)

        self.single_observation_space = spaces.Box(low=0, high=1,
            shape=(self.num_obs,), dtype=np.int8)
        self.single_action_space = spaces.Discrete(4)

        super().__init__(buf=buf)
        self.c_envs = CyTrashPickup(self.observations, self.actions, self.rewards, self.terminals, num_envs, num_agents, grid_size, num_trash, num_bins, max_steps, agent_sight_range)

    def reset(self, seed=None):
        self.c_envs.reset()
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_envs.step()
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = self.c_envs.log()
            # print(f"tha log: {log}")
            if log['episode_length'] > 0:
                info.append(log)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        self.c_envs.render()
        
    def close(self):
        self.c_envs.close() 

def test_performance(timeout=10, atn_cache=1024):
    env = TrashPickupEnv(num_envs=1024, grid_size=10, num_agents=4,
        num_trash=20, num_bins=1, max_steps=150, agent_sight_range=5)
 
    env.reset()
    tick = 0

    actions = np.random.randint(0, 4, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
