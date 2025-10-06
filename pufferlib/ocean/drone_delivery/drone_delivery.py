import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.drone_delivery import binding

class DroneDelivery(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        num_drones=64,

        box_base_density=50.0,
        box_k_growth=0.25,

        dist_decay=15.0,

        grip_k_decay=5.0,
        grip_k_max=17.105300970916993,
        grip_k_min=1.0,

        pos_const=0.7108647198043252,
        pos_penalty=0.0009629600280207475,

        reward_grip=0.9999,
        reward_ho_drop=0.20890470430909128,
        reward_hover=0.1603521816569735,

        reward_max_dist=65.0,
        reward_min_dist=0.9330297552248776,

        vel_penalty_clamp=0.25,

        w_approach=2.2462377881752698,
        w_position=0.7312628607193232,
        w_stability=1.6094417286807845,
        w_velocity=0.0009999999999999731,

        render_mode=None,
        report_interval=1024,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(45,),
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

                box_base_density=box_base_density,
                box_k_growth=box_k_growth,

                dist_decay=dist_decay,

                grip_k_decay=grip_k_decay,
                grip_k_max=grip_k_max,
                grip_k_min=grip_k_min,

                pos_const=pos_const,
                pos_penalty=pos_penalty,

                reward_grip=reward_grip,
                reward_ho_drop=reward_ho_drop,
                reward_hover=reward_hover,

                reward_max_dist=reward_max_dist,
                reward_min_dist=reward_min_dist,

                vel_penalty_clamp=vel_penalty_clamp,

                w_approach=w_approach,
                w_position=w_position,
                w_stability=w_stability,
                w_velocity=w_velocity
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
    env = DroneDelivery(num_envs=1000)
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
