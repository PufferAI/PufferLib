import numpy as np
import gymnasium
from .cy_continuous_cy import CContinuousCy

class ContinuousCyEnv(gymnasium.Env):
    def __init__(self, discretize=False):
        super().__init__()

        self.discretize = discretize
        self.c_env = CContinuousCy(discretize)
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        if discretize:
            self.action_space = gymnasium.spaces.Discrete(4)
        else:
            self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.render_mode = 'human'
        self.client = None

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        self.c_env.reset()
        state = self.c_env.get_state()
        return state, {}

    def step(self, action):
        state, reward, done, truncated, info = self.c_env.step(action)

        return state, reward, done, truncated, info

    def render(self):
        if self.client is None:
            self.client = RaylibClient()

        pos_x, pos_y, vel_x, vel_y, target_x, target_y = self.c_env.get_state()
        frame, action = self.client.render(pos_x, pos_y, target_x, target_y)

        return frame

    def close(self):
        pass


class RaylibClient:
    def __init__(self, width=1080, height=720, size=20):
        self.width = width
        self.height = height
        self.size = size

        from raylib import rl
        rl.InitWindow(width, height, "PufferLib Simple Continuous".encode())
        rl.SetTargetFPS(10)
        self.rl = rl

        from cffi import FFI
        self.ffi = FFI()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width * height * channels)
        return np.frombuffer(cdata, dtype=np.uint8).reshape((height, width, channels))[:, :, :3]

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

        pos_x = int((0.5 + pos_x / 2) * self.width)
        pos_y = int((0.5 + pos_y / 2) * self.height)
        target_x = int((0.5 + target_x / 2) * self.width)
        target_y = int((0.5 + target_y / 2) * self.height)

        rl.DrawCircle(pos_x, pos_y, self.size, [255, 0, 0, 255])
        rl.DrawCircle(target_x, target_y, self.size, [0, 0, 255, 255])

        rl.EndDrawing()

        return self._cdata_to_numpy(), action
    
def test_performance(discretize=False, atn_cache=1024, timeout=10):
    import time

    env = ContinuousCyEnv(discretize=discretize)
    env.reset()
    actions_cache = (np.random.randint(0, 4, atn_cache) if discretize 
                     else np.random.rand(atn_cache, 2).astype(np.float32))  # Convert to float32 here
    start = time.time()
    tick = 0
    while time.time() - start < timeout:
        action = actions_cache[tick % atn_cache]
        env.step(action)
        tick += 1
    elapsed_time = time.time() - start
    sps = tick / elapsed_time
    print(f"SPS: {sps:.2f}")

if __name__ == '__main__':
    test_performance()

