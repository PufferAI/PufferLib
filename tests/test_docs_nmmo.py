### Example PufferLib API usage with Neural MMO

import nmmo

import pufferlib
import pufferlib.emulation
import pufferlib.vectorization

# Wrap Neural MMO with PufferLib
def make_nmmo_env():
    return pufferlib.emulation.PettingZooPufferEnv(env_creator=nmmo.Env)

# Vectorize 4 environments across 2 cores
envs = pufferlib.vectorization.Serial(
    env_creator=make_nmmo_env, num_workers=2, envs_per_worker=2)

# Standard Gym/PettingZoo API. See custom CleanRL demo for async API.
obs = envs.reset(seed=42)
for _ in range(32):
    atns = [envs.single_action_space.sample()
        for _ in range(4*envs.num_agents)]

    obs, reward, done, info = envs.step(atns)