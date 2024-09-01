import numpy as np
import os

from cffi import FFI
from raylib import rl, colors
import pyray

PUFF_BACKGROUND = [6, 24, 24, 255]
PUFF_TEXT = [0, 187, 187, 255]

ANSI_COLORS = [30, 34, 36, 90, 31, 97, 91, 37]

COLORS = np.array([
    [6, 24, 24 ],     # Background
    [0, 0, 255],     # Food
    [0, 128, 255],   # Corpse
    [128, 128, 128], # Wall
    [255, 0, 0],     # Snake
    [255, 255, 255], # Snake
    [255, 85, 85],     # Snake
    [170, 170, 170], # Snake
], dtype=np.uint8)


def any_key_down(keys):
    for key in keys:
        if rl.IsKeyDown(key):
            return True
    return False

def any_key_pressed(keys):
    for key in keys:
        if rl.IsKeyPressed(key):
            return True
    return False

def cdata_to_numpy():
    image = rl.LoadImageFromScreen()
    data_pointer = image.data
    width = image.width
    height = image.height
    channels = 4
    data_size = width * height * channels
    cdata = FFI().buffer(data_pointer, data_size)
    return np.frombuffer(cdata, dtype=np.uint8
        ).reshape((height, width, channels))

def make_texture(width, height):
    rendered = np.zeros((height, width, 4), dtype=np.uint8)
    raylib_image = pyray.Image(FFI().from_buffer(rendered.data),
        width, height, 1, pyray.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
    return rl.LoadTextureFromImage(raylib_image)

class AnsiRender:
    def __init__(self, colors=None):
        self.colors = colors
        if colors is None:
            self.colors = ANSI_COLORS

    def render(self, grid):
        frame = ''
        for v in range(grid.shape[0]):
            lines = []
            for line in grid[v-1:-v, v-1:-v]:
                lines.append(''.join([
                    f'\033[{ANSI_COLORS[val]}m██\033[0m' for val in line]))

            frame = '\n'.join(lines)

        return frame
 
class RGBArrayRender:
    def __init__(self, colors=None, upscale=1):
        self.colors = colors
        if colors is None:
            self.colors = COLORS

        self.rescaler = np.ones((upscale, upscale, 1), dtype=np.uint8)
        self.upscale = upscale

    def render(self, grid):
        frame = self.colors[grid]

        if self.upscale > 1:
            frame = np.kron(frame, self.rescaler)

        return frame

class GridRender:
    def __init__(self, width, height, screen_width=1080, screen_height=720,
            colors=None, fps=60, name='PufferLib Raylib Renderer'):
        self.width = width
        self.height = height
        self.fps = fps

        self.colors = colors
        if colors is None:
            self.colors = COLORS

        rl.InitWindow(screen_width, screen_height, name.encode())
        rl.SetTargetFPS(fps)
        self.width = width
        self.height = height

        camera = pyray.Camera2D()
        camera.target= (width/2, height/2)
        camera.rotation = 0.0 
        camera.zoom = min(screen_width/width, screen_height/height)
        self.camera = camera

        self.speed = min(screen_width, screen_height) / 100
        self.texture = make_texture(width, height)

        self.show_help = False
        self.screen_width = screen_width
        self.screen_height = screen_height

    def render(self, grid, *args, end_drawing=True):
        assert grid.shape[0] == self.height
        assert grid.shape[1] == self.width
        rendered = self.colors[grid]

        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        screen_width = rl.GetScreenWidth()
        screen_height = rl.GetScreenHeight()

        camera = self.camera
        camera.offset.x = screen_width/2
        camera.offset.y = screen_height/2

        fps = rl.GetFPS() or self.fps
        fps_mul = self.fps / fps
        speed = self.speed * fps_mul
        zoom_speed = 0.01 * fps_mul

        if any_key_down([rl.KEY_SPACE]):
            camera.zoom = min(screen_width/self.width, screen_height/self.height)
            camera.target.x = self.width/2
            camera.target.y = self.height/2

        if any_key_down([rl.KEY_LEFT_SHIFT]):
            speed *= 3
            zoom_speed *= 3

        speed = speed / camera.zoom

        if any_key_down([rl.KEY_UP, rl.KEY_W]):
            camera.target.y -= speed
        if any_key_down([rl.KEY_DOWN, rl.KEY_S]):
            camera.target.y += speed
        if any_key_down([rl.KEY_LEFT, rl.KEY_A]):
            camera.target.x -= speed
        if any_key_down([rl.KEY_RIGHT, rl.KEY_D]):
            camera.target.x += speed
        if any_key_down([rl.KEY_Q, rl.KEY_MINUS]):
            camera.zoom /= 1 + zoom_speed
        if any_key_down([rl.KEY_E, rl.KEY_EQUAL]):
            camera.zoom *= 1 + zoom_speed

        if any_key_pressed([rl.KEY_TAB, rl.KEY_GRAVE]):
            self.show_help = not self.show_help

        rl.BeginDrawing()
        rl.BeginMode2D(self.camera)
        rl.ClearBackground(PUFF_BACKGROUND)
        rl.UpdateTexture(self.texture, rendered.tobytes())
        rl.DrawTextureEx(self.texture, (0, 0), 0, 1, colors.WHITE)
        rl.EndMode2D()
        if self.show_help:
            # Stats
            rl.DrawText(f'FPS: {fps}'.encode(), 10, 10, 20, PUFF_TEXT)
            rl.DrawText(f'Zoom: {camera.zoom:.2f}'.encode(), 10, 30, 20, PUFF_TEXT)
            rl.DrawText(f'X: {camera.offset.x:.2f}'.encode(), 10, 50, 20, PUFF_TEXT)
            rl.DrawText(f'Y: {camera.offset.y:.2f}'.encode(), 10, 70, 20, PUFF_TEXT)
            rl.DrawText(f'Speed: {speed:.2f}'.encode(), 10, 90, 20, PUFF_TEXT)

            # Controls
            rl.DrawText('Move: WASD/HJKL'.encode(), 10, 120, 20, PUFF_TEXT)
            rl.DrawText('Zoom: QE/-+'.encode(), 10, 140, 20, PUFF_TEXT)
            rl.DrawText('Turbo: Shift'.encode(), 10, 160, 20, PUFF_TEXT)
            rl.DrawText('Help: Tab/~'.encode(), 10, 180, 20, PUFF_TEXT)
            rl.DrawText('Reset: Space'.encode(), 10, 200, 20, PUFF_TEXT)

        if end_drawing:
            rl.EndDrawing()

        return cdata_to_numpy()

class GameRender:
    def __init__(self, width, height, screen_width=1080, screen_height=720,
            colors=None, name='PufferLib Raylib Game'):
        self.client = GridRender(width, height,
            screen_width, screen_height, colors, name)

    def render(self, grid, x, y):
        self.client.camera.target.x = x
        self.client.camera.target.y = y
        return self.client.render(grid)

class TestGameRender:
    def __init__(self, width, height, colors=None,
            tile_size=16, name='PufferLib Raylib Game'):
        assert width % tile_size == 0
        assert height % tile_size == 0
        assert (width // tile_size) % 2 == 1
        assert (height // tile_size) % 2 == 1

        self.width = width
        self.height = height

        self.colors = colors
        if colors is None:
            self.colors = COLORS

        self.x_tiles = width // tile_size
        self.y_tiles = height // tile_size

        rl.InitWindow(width, height, name.encode())
        rl.SetTargetFPS(60)

    def render(self, grid, agent_positions):
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
        rl.ClearBackground(PUFF_BACKGROUND)

        main_y, main_x = agent_positions[0]
        dx = self.x_tiles // 2
        dy = self.y_tiles // 2


if __name__ == '__main__':
    renderer = GridRender(256, 256)
    grid = np.random.randint(0, 3, (256, 256), dtype=np.uint8)
    while True:
        frame = renderer.render(grid)

