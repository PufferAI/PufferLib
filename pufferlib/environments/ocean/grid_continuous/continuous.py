import gymnasium
import numpy as np

class Continuous(gymnasium.Env):
    def __init__(self, discretize=False):
        self.observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(6,))
        self.discretize = discretize
        if discretize:
            self.action_space=gymnasium.spaces.Discrete(4)
        else:
            self.action_space=gymnasium.spaces.Box(
                low=-1, high=1, shape=(2,))

        self.render_mode = 'human'
        self.client = None

    def reset(self, seed=None, options=None):
        # pos_x, pos_y, vel_x, vel_y, target_x, target_y
        self.state = 2*np.random.rand(6)-1
        self.state[2:4] = 0
        self.tick = 0

        return self.state, {}

    def step(self, action):
        if self.discretize:
            accel_x, accel_y = 0, 0
            if action == 0:
                accel_x = -0.1
            elif action == 1:
                accel_x = 0.1
            elif action == 2:
                accel_y = -0.1
            elif action == 3:
                accel_y = 0.1
        else:
            accel_x, accel_y = 0.1*action

        self.state[2] += accel_x
        self.state[3] += accel_y
        self.state[0] += self.state[2]
        self.state[1] += self.state[3]

        pos_x, pos_y, vel_x, vel_y, target_x, target_y = self.state

        if pos_x < -1 or pos_x > 1 or pos_y < -1 or pos_y > 1:
            return self.state, -1, True, False, {'score': 0}

        dist = np.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2)
        reward = 0.02 * (1 - dist)

        self.tick += 1
        done = dist < 0.1
        truncated = self.tick >= 100

        # TODO: GAE implementation making agent not hit target
        # without a big reward here
        info = {}
        if done:
            reward = 5.0
            info = {'score': 1}
        elif truncated:
            reward = 0.0
            info = {'score': 0}

        return self.state, reward, done, truncated, info

    def render(self):
        if self.client is None:
            self.client = RaylibClient()

        pos_x, pos_y, vel_x, vel_y, target_x, target_y = self.state
        frame, atn = self.client.render(pos_x, pos_y, target_x, target_y)
        return frame

class RaylibClient:
    def __init__(self, width=1080, height=720, size=20):
        self.width = width
        self.height = height
        self.size = size

        from raylib import rl
        rl.InitWindow(width, height,
            "PufferLib Simple Continuous".encode())
        rl.SetTargetFPS(10)
        self.rl = rl

        from cffi import FFI
        self.ffi = FFI()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width*height*channels)
        return np.frombuffer(cdata, dtype=np.uint8
            ).reshape((height, width, channels))[:, :, :3]

    def render(self, pos_x, pos_y, target_x, target_y):
        rl = self.rl
        action = None
        if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_W):
            action = 0
        elif rl.IsKeyDown(rl.KEY_DOWN) or rl.IsKeyDown(rl.KEY_S):
            action = 1
        elif rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
            action = 2
        elif rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
            action = 3

        rl.BeginDrawing()
        rl.ClearBackground([6, 24, 24, 255])

        pos_x = int((0.5+pos_x/2) * self.width)
        pos_y = int((0.5+pos_y/2) * self.height)
        target_x = int((0.5+target_x/2) * self.width)
        target_y = int((0.5+target_y/2) * self.height)

        rl.DrawCircle(pos_x, pos_y, self.size, [255, 0, 0, 255])
        rl.DrawCircle(target_x, target_y, self.size, [0, 0, 255, 255])

        rl.EndDrawing()
        return self._cdata_to_numpy(), action


