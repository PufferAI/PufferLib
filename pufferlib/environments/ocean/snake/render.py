import numpy as np
import os


def make_renderer(width, height, asset_map=None,
        sprite_sheet_path=None, tile_size=16, render_mode='rgb_array'):
    if render_mode == 'human':
        return RaylibRender(width, height, asset_map,
            sprite_sheet_path, tile_size)
    else:
        return GridRender(width, height, asset_map)

class GridRender:
    def __init__(self, width, height, asset_map=None):
        self.width = width
        self.height = height
        self.asset_map = asset_map
        if asset_map is None:
            np.array([
                [255, 255, 255],
                [255, 0, 0],
                [0, 0, 0],
            ], dtype=np.uint8)

    def render(self, grid, *args, upscale=1):
        rendered = self.asset_map[grid]

        if upscale > 1:
            rescaler = np.ones((upscale, upscale, 1), dtype=np.uint8)
            rendered = np.kron(rendered, rescaler)

        return rendered

def _cdata_to_numpy(self):
    image = self.rl.LoadImageFromScreen()
    data_pointer = image.data
    width = image.width
    height = image.height
    channels = 4
    data_size = width * height * channels
    cdata = self.ffi.buffer(data_pointer, data_size)
    return np.frombuffer(cdata, dtype=np.uint8
        ).reshape((height, width, channels))

class RaylibLocal:
    def __init__(self, width, height, asset_map, tile_size=16):
        self.asset_map = asset_map

        from raylib import colors, rl
        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Snake".encode())
        rl.SetTargetFPS(15)
        self.colors = colors
        self.rl = rl
        self.tile_size = tile_size
        self.width = width
        self.height = height

        from cffi import FFI
        self.ffi = FFI()

    def render(self, grid, agent_positions, actions, vision_range=None):
        rl = self.rl
        if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_W):
            actions[0] = 0
        elif rl.IsKeyDown(rl.KEY_DOWN) or rl.IsKeyDown(rl.KEY_S):
            actions[0] = 1
        elif rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
            actions[0] = 2
        elif rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
            actions[0] = 3

        colors = self.colors
        rl.BeginDrawing()
        rl.ClearBackground((6, 24, 24))
        ts = self.tile_size

        main_r, main_c = agent_positions[0]
        for i, r in enumerate(range(main_r-self.height//2, main_r+self.height//2+1)):
            for j, c in enumerate(range(main_c-self.width//2, main_c+self.width//2+1)):
                if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
                    continue

                tile = grid[r, c]
                if tile == 0:
                    continue

                rl.DrawRectangle(j*ts, i*ts, ts, ts, self.asset_map[tile])

        rl.EndDrawing()
        return _cdata_to_numpy(self)[:, :, :3]

class RaylibGlobal:
    def __init__(self, width, height, asset_map, tile_size=16):
        self.asset_map = asset_map

        from raylib import colors, rl
        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Snake".encode())
        rl.SetTargetFPS(15)
        self.colors = colors
        self.rl = rl

        self.tile_size = tile_size
        self.width = width
        self.height = height

        from cffi import FFI
        self.ffi = FFI()

    def render(self, grid, agent_positions, actions, vision_range=None, main_vision=40):
        rl = self.rl
        colors = self.colors
        rl.BeginDrawing()
        rl.ClearBackground((6, 24, 24))
        ts = self.tile_size

        # Draw walls
        for r in range(self.height):
            for c in range(self.width):
                tile = grid[r, c]
                if tile == 0:
                    continue

                rl.DrawRectangle(
                    c*ts, r*ts, ts, ts, self.asset_map[tile])

        # Draw vision range
        if vision_range is not None:
            for r, c in agent_positions:
                xs = ts*(c - vision_range)
                ys = ts*(r - vision_range)
                xe = ts*(c + vision_range)
                ye = ts*(r + vision_range)
                rl.DrawRectangle(xs, ys, xe-xs, ye-ys, (255, 255, 255, 32))

        rl.EndDrawing()
        return _cdata_to_numpy(self)[:, :, :3]
