### Example PufferLib API usage with Neural MMO

import nmmo

import pufferlib
import pufferlib.emulation
import pufferlib.vectorization.serial

# Wrap Neural MMO with PufferLib
binding = pufferlib.emulation.Binding(
        env_cls=nmmo.Env,
        env_name='Neural MMO',
    )

# Vectorize 4 environments across 2 cores
envs = pufferlib.vectorization.serial.VecEnv(binding, num_workers=2, envs_per_worker=2)

# Standard Gym API. See custom CleanRL demo for async API.
obs = envs.reset(seed=42)
for _ in range(32):
    atns = [binding.single_action_space.sample() for _ in range(4*binding.max_agents)]
    obs, reward, done, info = envs.step(atns)