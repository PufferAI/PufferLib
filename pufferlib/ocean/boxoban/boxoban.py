
import os

import gymnasium
import numpy as np

import pufferlib
from pufferlib.ocean.boxoban import binding
from pufferlib.ocean.boxoban.parse_maps import write_bin

# MAP STUFF
BOXOBAN_DIR = os.path.dirname(__file__)
BOXOBAN_LEVELS = os.path.join(BOXOBAN_DIR, "boxoban-levels")
DIFFICULTY_SOURCES = {
    "basic": ("basic/train",),
    "easy": ("easy/train",),
    "medium": ("medium/train",),
    "hard": ("hard",),
    "unfiltered": ("unfiltered/train",),
}


def _collect_maps(difficulty):
    rel_paths = DIFFICULTY_SOURCES.get(difficulty)
    if rel_paths is None:
        raise ValueError(f"Invalid difficulty '{difficulty}'")

    maps = []
    for rel_path in rel_paths:
        level_dir = os.path.join(BOXOBAN_LEVELS, rel_path)
        if not os.path.isdir(level_dir):
            raise FileNotFoundError(f"Missing level directory {level_dir}")
        for filename in sorted(os.listdir(level_dir)):
            if filename.endswith(".txt"):
                maps.append(os.path.join(level_dir, filename))

    if not maps:
        raise RuntimeError(f"No map files found for difficulty '{difficulty}'")
    return maps


def _bin_path(difficulty):
    return os.path.join(BOXOBAN_DIR, f"boxoban_maps_{difficulty}.bin")


def _ensure_bin_exists(difficulty):
    path = _bin_path(difficulty)
    if not os.path.exists(path):
        maps = _collect_maps(difficulty)
        count = write_bin(maps, path, verbose=False)
        print(f"[Boxoban] Generated {count} puzzles for '{difficulty}' at {path}")
    return path

class Boxoban(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128, size=10, buf=None, seed=0, difficulty="basic", max_steps = 500,int_r_coeff = 0.1, target_loss_pen_coeff = 0.5, length_reward_coeff=0.1):
        self.shape = size*size*4 #agents walls boxes targets OHE
        difficulty = difficulty.lower()
        self.difficulty = difficulty
        self._difficulty_stat_key = f"difficulty ({self.difficulty})"

        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(self.shape,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(5)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.max_steps = max_steps
        self.int_r_coeff = int_r_coeff
        self.target_loss_pen_coeff = target_loss_pen_coeff
        self.length_reward_coeff = length_reward_coeff

        self.map_path = _ensure_bin_exists(self.difficulty)



        super().__init__(buf)
        self.c_envs = binding.vec_init(self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, size=size, max_steps = self.max_steps, int_r_coeff = self.int_r_coeff, target_loss_pen_coeff = self.target_loss_pen_coeff, map_path=self.map_path, length_reward_coeff = self.length_reward_coeff)
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1

        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            log_dict = binding.vec_log(self.c_envs)
            log_dict[self._difficulty_stat_key] = 1.0
            info.append(log_dict)

        return (self.observations, self.rewards,
            self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    N = 1

    env = Boxoban(num_envs=N)
    env.reset()
    env.render()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    i = 0
    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print('Boxoban SPS:', int(steps / (time.time() - start)))
