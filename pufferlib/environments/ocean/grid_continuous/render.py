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
        if asset_map is None:
            self.asset_map = {
                0: (255, 255, 255),
                1: (255, 0, 0),
                2: (0, 0, 0),
            }

    def render(self, grid, *args):
        rendered = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        for val in np.unique(grid):
            rendered[grid==val] = self.asset_map[val]

        return rendered

class RaylibRender:
    def __init__(self, width, height, asset_map=None,
            sprite_sheet_path=None, tile_size=16):
        '''Simple grid renderer for PufferLib grid environments'''
        if sprite_sheet_path is None:
            sprite_sheet_path = os.path.join(
                *self.__module__.split('.')[:-1], 'puffer-128-sprites.png')

        self.asset_map = None
        if asset_map is None:
            self.asset_map = {
                0: (0, 0, 128, 128),
                1: (0, 128, 128, 128),
                2: (128, 128, 128, 128),
                3: (0, 0, 128, 128),
                4: (128, 0, 128, 128),
            }

        from raylib import colors, rl
        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Grid".encode())
        rl.SetTargetFPS(60)
        self.colors = colors
        self.rl = rl

        self.puffer = rl.LoadTexture(sprite_sheet_path.encode())
        self.tile_size = tile_size
        self.width = width
        self.height = height

        from cffi import FFI
        self.ffi = FFI()

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
 
    def render(self, grid, agent_positions, actions, vision_range=None):
        colors = self.colors
        rl = self.rl

        rl.BeginDrawing()
        rl.ClearBackground((6, 24, 24))
        ts = self.tile_size

        # Draw walls
        for r in range(self.height):
            for c in range(self.width):
                if grid[r, c] == 2:
                    rl.DrawRectangle(
                        c*ts, r*ts, ts, ts, colors.BLACK)

        # Draw vision range
        if vision_range is not None:
            for r, c in agent_positions:
                xs = ts*(c - vision_range)
                ys = ts*(r - vision_range)
                xe = ts*(c + vision_range)
                ye = ts*(r + vision_range)
                rl.DrawRectangle(xs, ys, xe-xs, ye-ys, (255, 255, 255, 32))

        for idx, (r, c) in enumerate(agent_positions):
            if grid[r, c] == 1:
                atn = actions[idx]
                source_rect = self.asset_map[atn]
                dest_rect = (c*ts, r*ts, ts, ts)
                rl.DrawTexturePro(self.puffer, source_rect, dest_rect,
                    (0, 0), 0, colors.WHITE)
            elif grid[r, c] == 2:
                rl.DrawRectangle(
                    c*ts, r*ts, ts, ts, colors.BLACK)

        rl.EndDrawing()
        return self._cdata_to_numpy()[:, :, :3]
