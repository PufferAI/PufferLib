from pdb import set_trace as T
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from c_precompute_pathing import precompute_pathing as c_precompute_pathing

def precompute_pathing(grid):
    N = grid.shape[0]
    paths = np.zeros((N, N, N, N), dtype=np.uint8) + 255
    buffer = np.zeros((8*N*N, 3), dtype=np.int32)
    for r in range(N):
        for c in range(N):
            bfs(grid, paths[r, c], buffer, r, c)

    return paths

def bfs(grid, paths, buffer, dest_r, dest_c):
    N = grid.shape[0]
    if grid[dest_r, dest_c] == 1:
        return

    start = 0
    end = 1
    buffer[start, 0] = 0
    buffer[start, 1] = dest_r
    buffer[start, 2] = dest_c
    while start < end:
        atn = buffer[start, 0]
        start_r = buffer[start, 1]
        start_c = buffer[start, 2]
        start += 1

        if start_r < 0 or start_r >= N or start_c < 0 or start_c >= N:
            continue

        if paths[start_r, start_c] != 255:
            continue

        if grid[start_r, start_c] == 1:
            paths[start_r, start_c] = 8
            continue

        paths[start_r, start_c] = atn

        buffer[end, 0] = 0
        buffer[end, 1] = start_r - 1
        buffer[end, 2] = start_c
        end += 1

        buffer[end, 0] = 1
        buffer[end, 1] = start_r + 1
        buffer[end, 2] = start_c
        end += 1

        buffer[end, 0] = 2
        buffer[end, 1] = start_r
        buffer[end, 2] = start_c - 1
        end += 1

        buffer[end, 0] = 3
        buffer[end, 1] = start_r
        buffer[end, 2] = start_c + 1
        end += 1

        buffer[end, 0] = 4
        buffer[end, 1] = start_r - 1
        buffer[end, 2] = start_c + 1
        end += 1

        buffer[end, 0] = 5
        buffer[end, 1] = start_r + 1
        buffer[end, 2] = start_c + 1
        end += 1

        buffer[end, 0] = 6
        buffer[end, 1] = start_r + 1
        buffer[end, 2] = start_c - 1
        end += 1

        buffer[end, 0] = 7
        buffer[end, 1] = start_r - 1
        buffer[end, 2] = start_c - 1
        end += 1

    paths[dest_r, dest_c] = 8
    return end

from pufferlib.environments.ocean.render import GridRender, cdata_to_numpy

from raylib import rl

class RaylibVectorField:
    def __init__(self, grid, paths, tile_size=12):
        self.tile_size = tile_size
        self.grid = grid
        self.paths = paths

        width = grid.shape[1]
        height = grid.shape[0]
        self.tile_size = tile_size
        self.client = GridRender(
            width, height, tile_size*width, tile_size*height, fps=15)

    def render(self):
        self.client.render(self.grid, end_drawing=False)

        ts = self.tile_size
        R = self.grid.shape[0]
        C = self.grid.shape[1]

        pos = rl.GetMousePosition()
        end_r = int(pos.y // ts)
        end_c = int(pos.x // ts)

        rl.DrawRectangle(end_c*ts, end_r*ts, ts, ts, [255, 0, 0, 255])

        paths = self.paths[end_r, end_c]
        for r in range(R):
            for c in range(C):
                x = c*ts
                y = r*ts
                atn = paths[r, c]

                if atn == 1:
                    pos = [(x+ts//2, y), (x+ts//3, y+ts), (x+2*ts//3, y+ts)]
                elif atn == 0:
                    pos = [(x+2*ts//3, y), (x+ts//3, y), (x+ts//2, y+ts)]
                elif atn == 3:
                    pos = [(x, y+ts//2), (x+ts, y+2*ts//3), (x+ts, y+ts//3)]
                elif atn == 2:
                    pos = [(x, y+ts//3), (x, y+2*ts//3), (x+ts, y+ts//2)]
                elif atn == 4:
                    pos = [(x+2*ts//3, y), (x, y+ts), (x+ts, y+ts//3)]
                elif atn == 5:
                    pos = [(x, y), (x+2*ts//3, y+ts), (x+ts, y+2*ts//3)]
                elif atn == 6:
                    pos = [(x+ts, y), (x, y+2*ts//3), (x+ts//3, y+ts)]
                elif atn == 7:
                    pos = [(x+ts//3, y), (x, y+ts//3), (x+ts, y+ts)]
                else:
                    continue

                rl.DrawTriangle(pos[0], pos[1], pos[2], [0, 178, 178, 255])

        rl.EndDrawing()
        return cdata_to_numpy()


def test_bfs(grid, r1, c1):
    import time
    start = time.time()
    paths = c_precompute_pathing(grid)
    end = time.time()
    print(f'C Precompute Pathing: {end - start:.2f} seconds')
    renderer = RaylibVectorField(grid, paths)

    frames = []
    for i in range(90000000):
        frame = renderer.render()
        #frames.append(frame)
        #import PIL
        #PIL.Image.fromarray(frame).save('pathing.png')
        #exit(0)

    # Save frames as gif
    #import imageio
    #imageio.mimsave('pathing.gif', frames, fps=15, loop=0)


def test_performance(grid, r1, c1, timeout=10):
    N = grid.shape[0]
    paths = np.zeros((N, N), dtype=np.uint8) + 255
    buffer = np.zeros((8*N*N, 3), dtype=np.int32)

    import time
    start = time.time()
    iters = 0
    while time.time() - start < timeout:
        paths[:] = 255
        bfs(grid, paths, buffer, r1, c1)
        iters += 1

    start = time.time()
    c_iters = 0
    while time.time() - start < timeout:
        paths[:] = 255
        c_bfs(grid, paths, buffer, r1, c1)
        c_iters += 1

    print(f'BFS: {iters/timeout:.2f} SPS')
    print(f'C BFS: {c_iters/timeout:.2f} SPS')

if __name__ == '__main__':
    from PIL import Image
    game_map = np.array(Image.open('dota_map.png'))[:, :, -1]
    game_map = game_map[::2, ::2][1:-1, 1:-1]#[32:-32, 32:-32]
    game_map[game_map==0] = 1
    game_map[game_map==255] = 0
    test_bfs(game_map, 20, 128-16)
    #test_performance(game_map, 20, 128-16)
