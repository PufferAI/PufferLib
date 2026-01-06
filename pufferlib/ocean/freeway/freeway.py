import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.freeway import binding


class Freeway(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        frameskip=4,
        width=1216,
        height=720,
        player_width=64,
        player_height=64,
        car_width=64,
        car_height=40,
        lane_size=64,
        difficulty=0,
        level=0,
        use_dense_rewards=True,
        env_randomization=True,
        enable_human_player=False,
        log_interval=128,
        buf=None,
        seed=0,
    ):
        assert level < 8, "Level should be in {0, 1, 2, 3, 4, 5, 6, 7} or -1. Level -1 is a random mix of all 8 supported levels."
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(34,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.tick = 0

        self.single_action_space = gymnasium.spaces.Discrete(3)

        super().__init__(buf)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            frameskip=frameskip,
            width=width,
            height=height,
            player_width=player_width,
            player_height=player_height,
            car_width=car_width,
            car_height=car_height,
            lane_size=lane_size,
            difficulty=difficulty,
            level = level,
            enable_human_player=enable_human_player,
            env_randomization=env_randomization,
            use_dense_rewards=use_dense_rewards,
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
            
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=60, level = 0,atn_cache=1024):
    env = Freeway(num_envs=1024, level=level)
    env.reset()
    tick = 0

    actions = np.random.randint(0, 3, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1
    env.close()
    print(f'SPS: %f', env.num_agents * tick / (time.time() - start))

def test_render(timeout=60, level = 0,atn_cache=1024):
    env = Freeway(num_envs=1, level=level)
    env.reset(seed=0)
    tick = 0

    actions = np.random.randint(0, 3, (atn_cache, env.num_agents))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        obs, rew, term, trunc, i = env.step(atn)
        env.render()
        tick += 1
        if tick == 100:
            env.reset()
    env.close()


if __name__ == '__main__':
    # test_performance()
    for level in range(0,8):
        test_performance(level = level, timeout=5, atn_cache=1024)

