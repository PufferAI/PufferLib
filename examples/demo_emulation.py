from rich import print
from pufferlib.environments import minihack

make_env = minihack.env_creator()
env = make_env()

# Unwrapped environment
print('Raw environment:', env.env)
print('Raw observation space:', env.env.observation_space.keys())
print('Raw action space:', env.env.action_space)

breakpoint()
print()
print('---')
print()

# Wrapped environment
print('Puffer environment:', env)
print('Observation space:', env.observation_space.shape)
print('Observation space type:', env.observation_space.dtype)
print('Action space:', env.action_space)

breakpoint()
print()
print('---')
print()

# Emulation information
print('Emulation data')
print(env.emulated.observation_dtype)
print(env.emulated.emulated_observation_dtype)

breakpoint()
print()
print('---')
print()

# Real data
print('Data from environment')
flat_obs, _ = env.reset()

# View as structured, batched or unbatched
import pufferlib.emulation
struct_obs = flat_obs.view(env.emulated.emulated_observation_dtype)[0]
print('[red]blstats:', struct_obs['blstats'].shape, struct_obs['blstats'].dtype)
print('[red]chars:', struct_obs['chars'].shape, struct_obs['chars'].dtype)
print('[red]glyphs:', struct_obs['glyphs'].shape, struct_obs['glyphs'].dtype)


breakpoint()
print()
print('---')
print()

# Vectorized demo
import pufferlib.vector
vecenv = pufferlib.vector.make(make_env, num_envs=4)
flat_obs, _ = vecenv.reset()
struct_obs = flat_obs.view(env.emulated.emulated_observation_dtype)[:, 0]

print('Flat batch:', flat_obs.shape)
print('[red]blstats:', struct_obs['blstats'].shape, struct_obs['blstats'].dtype)
print('[red]chars:', struct_obs['chars'].shape, struct_obs['chars'].dtype)
print('[red]glyphs:', struct_obs['glyphs'].shape, struct_obs['glyphs'].dtype)
