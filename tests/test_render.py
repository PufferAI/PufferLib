from pdb import set_trace as T

import argparse
import importlib
import time

import cv2


# Tested human: classic_control, atari, minigrid
# Tested rbg_array: atari, pokemon_red, crafter
# Tested ansii: minihack, nethack, squared
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='atari')
    parser.add_argument('--render-mode', type=str, default='rgb_array')
    args = parser.parse_args()

    env_module = importlib.import_module(f'pufferlib.environments.{args.env}')

    if args.render_mode == 'human':
        env = env_module.make_env(render_mode='human')
    else:
        env = env_module.make_env()

    terminal = True
    while True:
        start = time.time()
        if terminal or truncated:
            ob, _ = env.reset()

        if args.render_mode == 'rgb_array':
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #if ob.shape[0] in (1, 3, 4):
            #    ob = ob.transpose(1, 2, 0)
            cv2.imshow('frame', frame)

            #cv2.imshow('ob', ob)
            cv2.waitKey(1)
        elif args.render_mode == 'ansi':
            chars = env.render()
            print("\033c", end="")
            print(chars)

        ob = ob.reshape(1, *ob.shape)
        action = env.action_space.sample()
        ob, reward, terminal, truncated, info = env.step(action)
        env.render()
        start = time.time()
        if time.time() - start < 1/60:
            time.sleep(1/60 - (time.time() - start))
           
