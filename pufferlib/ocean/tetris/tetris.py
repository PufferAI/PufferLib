import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.tetris import binding

class Tetris(pufferlib.PufferEnv):
    def __init__(
        self, 
        num_envs=1, 
        n_cols=10, 
        n_rows=20,
        use_deck_obs=True,
        n_noise_obs=10,
        n_init_garbage=4,
        render_mode=None, 
        log_interval=32,
        buf=None, 
        seed=0
    ):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(n_cols*n_rows + 6 + 7 * 4 + n_noise_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(7)
        self.render_mode = render_mode
        self.log_interval = log_interval
        self.num_agents = num_envs

        super().__init__(buf)
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            n_cols=n_cols,
            n_rows=n_rows,
            use_deck_obs=use_deck_obs,
            n_noise_obs=n_noise_obs,
            n_init_garbage=n_init_garbage,
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

if __name__ == '__main__':
    TIME = 10
    num_envs = 4096
    env = Tetris(num_envs=num_envs)
    actions = [
        [env.single_action_space.sample() for _ in range(num_envs) ]for _ in range(1000)
    ]
    obs, _ = env.reset(seed = np.random.randint(0,1000))

    import time
    start = time.time()
    tick = 0
    
    while time.time() - start < TIME:
        action = actions[tick%1000]
        env.render()
        print(np.array(obs[0][0:200]).reshape(20,10), obs[0][200:206], obs[0][206:])
        obs, _, _, _, _ = env.step(action)
        tick += 1
    print('SPS:', (tick*num_envs) / (time.time() - start))
    env.close()

