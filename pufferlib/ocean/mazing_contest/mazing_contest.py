import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.mazing_contest import binding

class MazingContest(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        build_time_limit=None,
        max_moves=None,
        max_rounds=None,
        min_gold=None,
        max_gold=None,
        min_lumber=None,
        max_lumber=None,
        report_interval=128,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(108,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Discrete(200)

        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.int32)

        kwargs = {}
        if build_time_limit is not None:
            kwargs['build_time_limit'] = build_time_limit
        if max_moves is not None:
            kwargs['max_moves'] = max_moves
        if max_rounds is not None:
            kwargs['max_rounds'] = max_rounds
        if min_gold is not None:
            kwargs['min_gold'] = min_gold
        if max_gold is not None:
            kwargs['max_gold'] = max_gold
        if min_lumber is not None:
            kwargs['min_lumber'] = min_lumber
        if max_lumber is not None:
            kwargs['max_lumber'] = max_lumber

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed,
            **kwargs
        )

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
    env = MazingContest(num_envs=1000)
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