import asyncio
import websockets
import json
import time
import colorsys
import gymnasium as gym
from pokegym import ram_map, game_map
import multiprocessing
from pokegym.bin.ram_reader.red_memory_battle import *
from pokegym.bin.ram_reader.red_memory_env import *
from pokegym.bin.ram_reader.red_memory_items import *
from pokegym.bin.ram_reader.red_memory_map import *
from pokegym.bin.ram_reader.red_memory_menus import *
from pokegym.bin.ram_reader.red_memory_player import *
from pokegym.bin.ram_reader.red_ram_api import *
from enum import IntEnum

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

def color_generator(step=1):
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
        self.color_generator = color_generator(step=1) 
        self.ws_address = "wss://poke-ws-test-ulsjzjzwpa-ue.a.run.app/broadcast"
        self.stream_metadata = stream_metadata
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = self.loop.run_until_complete(
                self.establish_wc_connection()
        )
        self.upload_interval = 100
        self.stream_step_counter = 0
        self.coord_list = []
        self.start_time = time.time()
        self.api = Game(env.game)
        self.api.process_game_states()
        self.bet = '''
BBBB   EEEE  TTTTT
B   B  E       T  
BBBB   EEEE    T  
B   B  E       T  
BBBB   EEEE    T  
'''
        self.fb = '''
01000110 01000010
01010010 01001001
01000101 01001100
01000101 01001100 
'''

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
        game_state = self.env.api.game_state.name
        get_gs = self.api.get_game_state()
        x_pos = self.emulator.get_memory_value(X_POS_ADDRESS)
        y_pos = self.emulator.get_memory_value(Y_POS_ADDRESS)
        map_n = self.emulator.get_memory_value(MAP_N_ADDRESS)
        self.coord_list.append([x_pos, y_pos, map_n])
        # next_color = next(self.color_generator)
        # color_as_rgb = [int(c * 255) for c in next_color]  # Convert to RGB values
        # self.stream_metadata["color"] = color_as_rgb
        # self.stream_metadata["extra"] = f"{self.bet}"
        self.stream_metadata["extra"] = f"{game_state}\nuptime: {round(self.uptime(), 2)} min"
        self.stream_metadata["color"] = next(self.color_generator)
        # self.stream_metadata["color"] = color_generator()
        # self.stream_metadata["env_id"] = f"{self.fb}{self.bet}"
        # self.stream_metadata["user"] = f"{self.get_colored_bet()}"

        if self.stream_step_counter >= self.upload_interval:
            # self.stream_metadata["extra"] = f"{game_state} | uptime: {round(self.uptime(), 2)} min."
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
            self.stream_step_counter = 0
            self.coord_list = []

        self.stream_step_counter += 1

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
    
    def uptime(self):
        return (time.time() - self.start_time) / 60


# import asyncio
# import websockets
# import json

# import gymnasium as gym

# X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
# MAP_N_ADDRESS = 0xD35E



# class StreamWrapper(gym.Wrapper):
#     def __init__(self, env, stream_metadata={}):
#         super().__init__(env)
#         self.ws_address = "wss://poke-ws-test-ulsjzjzwpa-ue.a.run.app/broadcast"
#         self.stream_metadata = stream_metadata
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_until_complete(self.connect_with_retry())

#         # self.websocket = self.loop.run_until_complete(
#         #         websockets.connect(self.ws_address)
#         # )
#         self.upload_interval = 200
#         self.stream_step_counter = 0
#         self.coord_list = []
#         if hasattr(env, "pyboy"):
#             self.emulator = env.pyboy
#         elif hasattr(env, "game"):
#             self.emulator = env.game
#         else:
#             raise Exception("Could not find emulator!")

#     async def connect_with_retry(self):
#         while True:
#             try:
#                 # Attempt to connect with a timeout of 10 seconds
#                 self.websocket = await asyncio.wait_for(
#                     websockets.connect(self.ws_address), timeout=10)
#                 break  # Connected successfully, exit the loop
#             except asyncio.TimeoutError:
#                 print("Connection attempt timed out, retrying...")
#                 await asyncio.sleep(5)  # Wait for a while before retrying
#             except websockets.exceptions.WebSocketException as e:
#                 print(f"WebSocket connection error: {e}")
#                 await asyncio.sleep(5)

    
#     def step(self, action):

#         x_pos = self.emulator.get_memory_value(X_POS_ADDRESS)
#         y_pos = self.emulator.get_memory_value(Y_POS_ADDRESS)
#         map_n = self.emulator.get_memory_value(MAP_N_ADDRESS)
#         self.coord_list.append([x_pos, y_pos, map_n])

#         if self.stream_step_counter >= self.upload_interval:
#             self.loop.run_until_complete(
#                 self.broadcast_ws_message(
#                     json.dumps(
#                         {
#                           "metadata": self.stream_metadata,
#                           "coords": self.coord_list
#                         }
#                     )
#                 )
#             )
#             self.stream_step_counter = 0
#             self.coord_list = []

#         self.stream_step_counter += 1

#         return self.env.step(action)

#     async def broadcast_ws_message(self, message):
#         try:
#             await self.websocket.send(message)
#         except (asyncio.TimeoutError, asyncio.exceptions.CancelledError) as e:
#             print(f"Connection error: {e}")
#             # Start reconnection attempt in a separate task
#             asyncio.create_task(self.connect_with_retry())
#         except websockets.exceptions.WebSocketException as e:
#             print(f"WebSocket send error: {e}")
#             # Handle other specified errors here
#             if isinstance(e, (
#                 websockets.exceptions.ConnectionClosed,
#                 websockets.exceptions.ConnectionClosedError,
#                 websockets.exceptions.ConnectionClosedOK,
#                 websockets.exceptions.InvalidHandshake,
#                 websockets.exceptions.SecurityError,
#                 websockets.exceptions.InvalidMessage,
#                 websockets.exceptions.InvalidHeader,
#                 websockets.exceptions.InvalidHeaderFormat,
#                 websockets.exceptions.InvalidHeaderValue,
#                 websockets.exceptions.InvalidOrigin,
#                 websockets.exceptions.InvalidUpgrade,
#                 websockets.exceptions.InvalidStatus,
#                 websockets.exceptions.InvalidStatusCode,
#                 websockets.exceptions.NegotiationError,
#                 websockets.exceptions.DuplicateParameter,
#                 websockets.exceptions.InvalidParameterName,
#                 websockets.exceptions.InvalidParameterValue,
#                 websockets.exceptions.AbortHandshake,
#                 websockets.exceptions.RedirectHandshake,
#                 websockets.exceptions.InvalidState,
#                 websockets.exceptions.InvalidURI,
#                 websockets.exceptions.PayloadTooBig,
#                 websockets.exceptions.ProtocolError,
#                 websockets.exceptions.WebSocketProtocolError,
#             )):
#                 print(f"WebSocket error: {e}")
#                 # Start reconnection attempt in a separate task
#                 asyncio.create_task(self.connect_with_retry())
#             else:
#                 # Handle other unspecified WebSocket exceptions
#                 pass 
        
    
#     async def broadcast_ws_message(self, message):
#         try:
#             while True:
#                 await self.websocket.send(message)
#         except websockets.exceptions.WebSocketException as e:
#             # attempt reconnection after a timeout or error
#             self.websocket = await websockets.connect(self.ws_address)