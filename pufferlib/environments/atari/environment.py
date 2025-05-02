from pdb import set_trace as T
import numpy as np
import functools

import gymnasium as gym

import pufferlib
import pufferlib.emulation
import pufferlib.environments
import pufferlib.utils
import pufferlib.postprocess
import pufferlib.postprocess

def env_creator(name='breakout'):
    return functools.partial(make, name)

def make(name, obs_type='grayscale', frameskip=4,
        full_action_space=False, framestack=1,
        repeat_action_probability=0.0, render_mode='rgb_array',
        buf=None, seed=0):
    '''Atari creation function'''
    pufferlib.environments.try_import('ale_py', 'AtariEnv')

    ale_render_mode = render_mode
    if render_mode == 'human':
        ale_render_mode = 'rgb_array'
        obs_type = 'rgb'
        frameskip = 1
        full_action_space = True
        upscale = 4
    elif render_mode == 'raylib':
        ale_render_mode = 'rgb_array'
        upscale = 8

    from ale_py import AtariEnv
    env = AtariEnv(name, obs_type=obs_type, frameskip=frameskip,
        repeat_action_probability=repeat_action_probability,
        full_action_space=full_action_space,
        render_mode=ale_render_mode)

    action_set = env._action_set
                    
    if render_mode != 'human':
        env = pufferlib.postprocess.ResizeObservation(env, downscale=2)

    if framestack > 1:
        env = gym.wrappers.FrameStack(env, framestack)

    if render_mode in ('human', 'raylib'):
        env = RaylibClient(env, action_set, frameskip, upscale)
    else:
        env = AtariPostprocessor(env) # Don't use standard postprocessor

    env = pufferlib.postprocess.EpisodeStats(env)
    env = pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
    return env

class AtariPostprocessor(gym.Wrapper):
    '''Atari breaks the normal PufferLib postprocessor because
    it sends terminal=True every live, not every episode'''
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space.shape
        if len(shape) < 3:
            shape = (1, *shape)
        else:
            shape = (shape[2], shape[0], shape[1])

        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=shape, dtype=env.observation_space.dtype)

    def unsqueeze_transpose(self, obs):
        if len(obs.shape) == 3:
            return np.transpose(obs, (2, 0, 1))
        else:
            return np.expand_dims(obs, 0)

    def reset(self, seed=None, options=None):
        obs, _ = self.env.reset(seed=seed)
        return self.unsqueeze_transpose(obs), {}

    def step(self, action):
        obs, reward, terminal, truncated, _ = self.env.step(action)
        return self.unsqueeze_transpose(obs), reward, terminal, truncated, {}

class RaylibClient(gym.Wrapper):
    def __init__(self, env, action_set, frameskip=4, upscale=4):
        self.env = env

        self.keymap = {}
        for i, atn in enumerate(action_set):
            self.keymap[atn.value] = i

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2:
            height, width = obs_shape
            channels = 1
        else:
            height, width, channels = obs_shape

        height *= upscale
        width *= upscale
        from raylib import rl, colors
        rl.InitWindow(width, height, "Atari".encode())
        rl.SetTargetFPS(60//frameskip)
        self.rl = rl
        self.colors = colors


        import numpy as np
        rendered = np.zeros((width, height, 4), dtype=np.uint8)

        import pyray
        from cffi import FFI
        raylib_image = pyray.Image(FFI().from_buffer(rendered.data),
            width, height, 1, pyray.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
        self.texture = rl.LoadTextureFromImage(raylib_image)
        self.action = 0

        self.upscale = upscale
        self.rescaler = np.ones((upscale, upscale, 1), dtype=np.uint8)

    def any_key_pressed(self, keys):
        for key in keys:
            if self.rl.IsKeyPressed(key):
                return True
        return False

    def any_key_down(self, keys):
        for key in keys:
            if self.rl.IsKeyDown(key):
                return True
        return False

    def down(self):
        return self.any_key_down([self.rl.KEY_S, self.rl.KEY_DOWN])

    def up(self):
        return self.any_key_down([self.rl.KEY_W, self.rl.KEY_UP])

    def left(self):
        return self.any_key_down([self.rl.KEY_A, self.rl.KEY_LEFT])

    def right(self):
        return self.any_key_down([self.rl.KEY_D, self.rl.KEY_RIGHT])

    def render(self):
        from ale_py import Action

        rl = self.rl
        if rl.IsKeyPressed(rl.KEY_ESCAPE):
            exit(0)

        elif rl.IsKeyDown(rl.KEY_SPACE):
            if self.left() and self.down():
                action = Action.DOWNLEFTFIRE.value
            elif self.right() and self.down():
                action = Action.DOWNRIGHTFIRE.value
            elif self.left() and self.up():
                action = Action.UPLEFTFIRE.value
            elif self.right() and self.up():
                action = Action.UPRIGHTFIRE.value
            elif self.left():
                action = Action.LEFTFIRE.value
            elif self.right():
                action = Action.RIGHTFIRE.value
            elif self.up():
                action = Action.UPFIRE.value
            elif self.down():
                action = Action.DOWNFIRE.value
            else:
                action = Action.FIRE.value
        elif self.left() and self.down():
            action = Action.DOWNLEFT.value
        elif self.right() and self.down():
            action = Action.DOWNRIGHT.value
        elif self.left() and self.up():
            action = Action.UPLEFT.value
        elif self.right() and self.up():
            action = Action.UPRIGHT.value
        elif self.left():
            action = Action.LEFT.value
        elif self.right():
            action = Action.RIGHT.value
        elif self.up():
            action = Action.UP.value
        else:
            action = Action.NOOP.value

        if action in self.keymap:
            self.action = self.keymap[action]
        else:
            self.action = Action.NOOP.value

        #frame = self.env.render()
        frame = self.frame
        if len(frame.shape) < 3:
            frame = np.expand_dims(frame, 2)
            frame = np.repeat(frame, 3, axis=2)

        if self.upscale > 1:
            frame = np.kron(frame, self.rescaler)

        rl.BeginDrawing()
        rl.ClearBackground(self.colors.BLACK)
        rl.UpdateTexture(self.texture, frame.tobytes())
        rl.DrawTextureEx(self.texture, (0, 0), 0, 1, self.colors.WHITE)
        rl.EndDrawing()

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.frame = obs
        return obs, info

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(self.action)
        self.frame = obs
        return obs, reward, terminal, truncated, info
