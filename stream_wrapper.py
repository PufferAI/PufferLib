import asyncio
import websockets
import json

import gymnasium as gym
import colorsys
import time

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E


def color_generator(step=5): # step=1
    """Generates a continuous spectrum of colors in hex format."""
    hue = 0
    while True:
        # Convert HSL (Hue, Saturation, Lightness) to RGB, then to HEX
        rgb = colorsys.hls_to_rgb(hue / 360, 0.5, 1)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        yield hex_color
        hue = (hue + step) % 360

def colors_generator(step=1):
    """Generates a continuous spectrum of colors in hex format."""
    hue = 0
    while True:
        rgb = colorsys.hls_to_rgb(hue / 360, 0.5, 1)
        yield rgb
        hue = (hue + step) % 360


class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.color_generator = color_generator(step=2) # step=1
        # self.ws_address = "wss://poke-ws-test-ulsjzjzwpa-ue.a.run.app/broadcast"
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = stream_metadata
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(
            self.establish_wc_connection()
        )
        self.upload_interval = 125
        self.steam_step_counter = 0
        self.coord_list = []
        self.start_time = time.time()        
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")
        
    def get_colored_bet(self):
        inner_colors = [next(self.color_generator) for _ in range(3)]  # Three inner squares
        outer_color = next(self.color_generator)

        def colorize_square(square, color):
            return f'\033[38;2;{int(color[0]*255)};{int(color[1]*255)};{int(color[2]*255)}m{square}\033[0m'

        colored_bet = self.bet
        for i, color in enumerate(inner_colors, start=1):
            colored_bet = colored_bet.replace(f'█{i}', colorize_square(f'█{i}', color))

        colored_bet = colored_bet.replace('█', colorize_square('█', outer_color))  # Colorize outer squares
        return colored_bet

    def step(self, action):

        x_pos = self.emulator.get_memory_value(X_POS_ADDRESS)
        y_pos = self.emulator.get_memory_value(Y_POS_ADDRESS)
        map_n = self.emulator.get_memory_value(MAP_N_ADDRESS)
        reset_count = self.env.reset_count
        env_id = self.env.env_id
        self.coord_list.append([x_pos, y_pos, map_n])
        
        
        self.stream_metadata["extra"] = f"uptime: {round(self.uptime(), 2)} min, reset#: {reset_count}, {env_id}"
        self.stream_metadata["color"] = next(self.color_generator)
        
        if self.steam_step_counter >= self.upload_interval:
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps(
                        {
                          "metadata": self.stream_metadata,
                          "coords": self.coord_list
                        }
                    )
                )
            )
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await self.websocket.send(message)
            except websockets.exceptions.WebSocketException as e:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await websockets.connect(self.ws_address)
        except:
            self.websocket = None

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    
    def uptime(self):
        return (time.time() - self.start_time) / 60
