### Example PufferLib API usage with Neural MMO

import nmmo

import pufferlib
import pufferlib.emulation
import vectorization.multiprocessing

# Wrap Neural MMO with PufferLib
binding = pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        env_name='Neural MMO',
    )

# Vectorize 4 environments across 2 cores
envs = pufferlib.vectorization.RayVecEnv(binding, num_workers=2, envs_per_worker=2)
envs.seed(42)

# Standard Gym API. See custom CleanRL demo for async API.
obs = envs.reset()
for _ in range(32):
    atns = [binding.single_action_space.sample() for _ in range(4*binding.max_agents)]
    obs, reward, done, info = envs.step(atns)