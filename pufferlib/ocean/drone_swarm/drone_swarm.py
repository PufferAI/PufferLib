import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.drone_swarm import binding

class DroneSwarm(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        num_drones=64,
        max_rings=5,
        render_mode=None,
        report_interval=1024,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(41,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        self.num_agents = num_envs*num_drones
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)

        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i*num_drones:(i+1)*num_drones],
                self.actions[i*num_drones:(i+1)*num_drones],
                self.rewards[i*num_drones:(i+1)*num_drones],
                self.terminals[i*num_drones:(i+1)*num_drones],
                self.truncations[i*num_drones:(i+1)*num_drones],
                i,
                num_agents=num_drones,
                max_rings=max_rings,
            ))

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = DroneSwarm(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")

if __name__ == "__main__":
    test_performance()
