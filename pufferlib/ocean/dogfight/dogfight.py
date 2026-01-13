import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.dogfight import binding


class Dogfight(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        render_mode=None,
        report_interval=1,
        buf=None,
        seed=42,
        max_steps=3000,
    ):
        # player(13) + rel_pos(3) + rel_vel(3) = 19
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(19,),
            dtype=np.float32,
        )

        # Action: Box(5) continuous [-1, 1]
        # [0] throttle, [1] elevator, [2] ailerons, [3] rudder, [4] trigger
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32
        )

        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)  # REQUIRED for continuous

        c_envs = []
        for env_num in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[env_num:(env_num+1)],
                self.actions[env_num:(env_num+1)],
                self.rewards[env_num:(env_num+1)],
                self.terminals[env_num:(env_num+1)],
                self.truncations[env_num:(env_num+1)],
                env_num,
                report_interval=self.report_interval,
                max_steps=max_steps,
            ))

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed else 0)
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
    env = Dogfight(num_envs=1000)
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
