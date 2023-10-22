from pufferlib.environments import PokemonRed

env = PokemonRed()
ob, info = env.reset()
for i in range(100):
    ob, reward, terminal, truncated, info = env.step(env.action_space.sample())
    print(f'Step: {i}, Info: {info}')
