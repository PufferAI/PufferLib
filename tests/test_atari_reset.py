from pdb import set_trace as T
from pufferlib.registry import atari

from cleanrl_ppo_atari import make_env


#env = make_env('BreakoutNoFrameskip-v4', 1, None, None, None)()
env = atari.make_env('BreakoutNoFrameskip-v4', 4)

obs = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f'i: {i}, info: {info["lives"]}')
    if done:
        obs = env.reset()
