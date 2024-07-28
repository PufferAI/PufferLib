# Minimal SB3 demo using PufferLib's environment wrappers

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from pufferlib.environments import atari

'''
    elif args.backend == 'sb3':
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.env_util import make_vec_env
        from sb3_contrib import RecurrentPPO

        envs = make_vec_env(lambda: make_env(**args.env),
            n_envs=args.train.num_envs, seed=args.train.seed, vec_env_cls=DummyVecEnv)

        model = RecurrentPPO("CnnLstmPolicy", envs, verbose=1,
            n_steps=args.train.batch_rows*args.train.bptt_horizon,
            batch_size=args.train.batch_size, n_epochs=args.train.update_epochs,
            gamma=args.train.gamma
        )

        model.learn(total_timesteps=args.train.total_timesteps)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
args = parser.parse_args()

env_creator = atari.env_creator(args.env)
envs = make_vec_env(lambda: env_creator(),
    n_envs=4, seed=0, vec_env_cls=DummyVecEnv)

model = PPO("CnnPolicy", envs, verbose=1)
model.learn(total_timesteps=2000)

# Demonstrate loading
model.save(f'ppo_{args.env}')
model = PPO.load(f'ppo_{args.env}')

# Watch the agent play
env = atari.make_env(args.env, render_mode='human')
terminal = True
for _ in range(1000):
    if terminal or truncated:
        ob, _ = env.reset()

    ob = ob.reshape(1, *ob.shape)
    action, _states = model.predict(ob)
    ob, reward, terminal, truncated, info = env.step(action[0])
    env.render()
       
