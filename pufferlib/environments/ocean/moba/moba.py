from pdb import set_trace as T
import numpy as np
import os

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.environments.ocean.moba.c_moba import Environment as CEnv
from pufferlib.environments.ocean.moba.c_moba import entity_dtype

EMPTY = 0
FOOD = 1
WALL = 2
AGENT_1 = 3
AGENT_2 = 4
AGENT_3 = 5
AGENT_4 = 6

PASS = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4

COLORS = np.array([
    [6, 24, 24, 255],     # Background
    [0, 0, 255, 255],     # Food
    [0, 128, 255, 255],   # Corpse
    [128, 128, 128, 255], # Wall
    [255, 0, 0, 255],     # Snake
    [255, 255, 255, 255], # Snake
    [255, 85, 85, 255],     # Snake
    [170, 170, 170, 255], # Snake
], dtype=np.uint8)


class PufferMoba(pufferlib.PufferEnv):
    def __init__(self, vision_range=5, agent_speed=0.25,
            discretize=False, report_interval=32, render_mode='rgb_array'):
        super().__init__()

        self.height = 128
        self.width = 128
        self.num_agents = 10
        self.num_towers = 18
        self.num_creeps = 100
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.report_interval = report_interval

        self.obs_size = 2*self.vision_range + 1

        # load game map from png
        game_map_path = os.path.join(
            *self.__module__.split('.')[:-1], 'dota_bitmap.png')
        from PIL import Image
        game_map = np.array(Image.open(game_map_path))[:, :, -1]

        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.grid[game_map != 0] = WALL

        self.pids = np.zeros((self.height, self.width), dtype=np.int32) - 1

        dtype = entity_dtype()
        self.c_entities = np.zeros(
            self.num_agents + self.num_towers + self.num_creeps, dtype=dtype)
        self.entities = self.c_entities.view(np.recarray)
        self.entities.pid = -1
        self.c_obs_players = np.zeros((self.num_agents, 10), dtype=dtype)
        self.obs_players = self.c_obs_players.view(np.recarray)

        self.emulated = None

        self.buf = pufferlib.namespace(
            observations = np.zeros(
                self.num_agents*self.obs_size*self.obs_size + 3, dtype=np.uint8),
            rewards = np.zeros(self.num_agents, dtype=np.float32),
            terminals = np.zeros(self.num_agents, dtype=bool),
            truncations = np.zeros(self.num_agents, dtype=bool),
            masks = np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros((self.num_agents, 3), dtype=np.uint32)

        self.render_mode = render_mode
        if render_mode == 'human':
            COLORS = np.array([
                [6, 24, 24, 255],     # Background
                [0, 0, 255, 255],     # Food
                [0, 128, 255, 255],   # Corpse
                [128, 128, 128, 255], # Wall
                [255, 0, 0, 255],     # Snake
                [255, 255, 255, 255], # Snake
                [255, 85, 85, 255],     # Snake
                [170, 170, 170, 255], # Snake
            ], dtype=np.uint8)

            self.client = RaylibClient(41, 23, COLORS.tolist())
     
        self.observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size+3,), dtype=np.uint8)

        if discretize:
            self.action_space = gymnasium.spaces.MultiDiscrete([3, 3, 10, 2, 2])
        else:
            finfo = np.finfo(np.float32)
            self.action_space = gymnasium.spaces.Box(
                low=finfo.min,
                high=finfo.max,
                shape=(5,),
                dtype=np.float32
            )

        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.cenv = None
        self.done = True
        self.infos = {}

    def render(self, upscale=4):
        grid = self.grid
        if self.render_mode == 'rgb_array':
            v = self.vision_range
            frame = COLORS[grid[v:-v-1, v:-v-1]]

            if upscale > 1:
                rescaler = np.ones((upscale, upscale, 1), dtype=np.uint8)
                frame = np.kron(frame, rescaler)

            return frame

        frame, self.human_action = self.client.render(
            self.grid, self.pids, self.entities, self.obs_players,
            self.actions, self.discretize)
        return frame

    def _fill_observations(self):
        self.buf.observations[:, -3] = (
            255*self.entities[:self.num_agents].y/self.height).astype(np.uint8)
        self.buf.observations[:, -2] = (
            255*self.entities[:self.num_agents].x/self.width).astype(np.uint8)
        self.buf.observations[:, -1] = (255*self.buf.rewards).astype(np.uint8)

    def reset(self, seed=0):
        if self.cenv is None:
            self.obs_view = self.buf.observations[:, 
                :self.obs_size*self.obs_size].reshape(
                self.num_agents, self.obs_size, self.obs_size)
            
        self.agents = [i+1 for i in range(self.num_agents)]
        self.done = False
        self.tick = 1

        self.grid[
            self.entities.y.astype(np.int32),
            self.entities.x.astype(np.int32)
        ] = AGENT_1

        self.cenv = CEnv(self.grid, self.pids, self.c_entities, self.c_obs_players,
            self.obs_view, self.buf.rewards, self.num_agents, self.num_creeps,
            self.num_towers, self.vision_range, self.agent_speed, self.discretize)
        self.cenv.reset()

        self.sum_rewards = []
        self._fill_observations()
        return self.buf.observations, self.infos

    def step(self, actions):
        if self.render_mode == 'human' and self.human_action is not None:
            #print(self.human_action)
            actions[0] = self.human_action

        if self.discretize:
            actions = actions.astype(np.uint32)
        else:
            actions = np.clip(actions,
                np.array([-1, -1, 0, 0, 0]),
                np.array([1, 1, 10, 1, 1])
            ).astype(np.float32)

        self.buf.rewards.fill(0)
        self.actions = actions
        self.cenv.step(actions)

        infos = self.infos
        self.sum_rewards.append(self.buf.rewards.sum())

        self.tick += 1
        if self.tick % self.report_interval == 0:
            infos['reward'] = np.mean(self.sum_rewards) / self.num_agents
            self.sum_rewards = []

        self._fill_observations()
        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, infos)

class RaylibClient:
    def __init__(self, width, height, asset_map, tile_size=32):
        self.width = width
        self.height = height
        self.asset_map = asset_map
        self.tile_size = tile_size

        sprite_sheet_path = os.path.join(
            *self.__module__.split('.')[:-1], 'puffer_chars.png')
        self.asset_map = {
            3: (0, 0, 128, 128),
            4: (128, 0, 128, 128),
            5: (256, 0, 128, 128),
            6: (384, 0, 128, 128),
            1: (512, 0, 128, 128),
        }

        from raylib import rl, colors
        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Grid".encode())
        rl.SetTargetFPS(20)
        self.puffer = rl.LoadTexture(sprite_sheet_path.encode())
        self.rl = rl
        self.colors = colors

        import pyray as ray
        camera = ray.Camera2D()
        camera.target = ray.Vector2(0.0, 0.0)
        camera.rotation = 0.0 
        camera.zoom = 1.0
        self.camera = camera

        from cffi import FFI
        self.ffi = FFI()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width*height*channels)
        return np.frombuffer(cdata, dtype=np.uint8
            ).reshape((height, width, channels))[:, :, :3]

    def render(self, grid, pids, entities, obs_players, actions, discretize):
        rl = self.rl
        colors = self.colors
        ay, ax = None, None
        if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_W):
            ay = 0 if discretize else -1
        if rl.IsKeyDown(rl.KEY_DOWN) or rl.IsKeyDown(rl.KEY_S):
            ay = 2 if discretize else 1
        if rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
            ax = 0 if discretize else -1
        if rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
            ax = 2 if discretize else 1

        skill_attack = 0
        skill_heal = 0
        if rl.IsKeyDown(rl.KEY_Q):
            skill_attack = 1
        if rl.IsKeyDown(rl.KEY_W):
            skill_heal = 1

        ts = self.tile_size
        main_r = entities[0].y
        main_c = entities[0].x
        self.camera.target.x = (main_c - self.width//2) * ts
        self.camera.target.y = (main_r - self.height//2) * ts
        main_r = int(main_r)
        main_c = int(main_c)
        r_min = main_r - self.height//2 - 1
        r_max = main_r + self.height//2 + 1
        c_min = main_c - self.width//2 - 1
        c_max = main_c + self.width//2 + 1

        pos = rl.GetMousePosition()
        raw_mouse_x = pos.x + self.camera.target.x
        raw_mouse_y = pos.y + self.camera.target.y
        mouse_x = int(raw_mouse_x // ts)
        mouse_y = int(raw_mouse_y // ts)
        #mouse_x = int(c_min + pos.x // ts + 1)
        #mouse_y = int(r_min + pos.y // ts + 1)

        ay = np.clip((pos.y - ts*self.height//2) / 200, -1, 1)
        ax = np.clip((pos.x - ts*self.width//2) / 200, -1, 1)
        #print(f'Mouse: {pos.x}, {pos.y}, Target: {ts*mouse_x}, {ts*mouse_y}, Action: {ay}, {ax}')

        target_pid = pids[mouse_y, mouse_x]
        #print(f'Mouse: {mouse_x}, {mouse_y}, Target: {target_pid}')
        attack = 0
        if target_pid == -1:
            attack = 0
        else:
            target = entities[target_pid]
            for i in range(10):
                if obs_players[0, i].pid == target_pid:
                    attack = i
                    break

        if ax is None and ay is None:
            action = None
        else:
            if ax is None:
                ax = 1 if discretize else 0
            if ay is None:
                ay = 1 if discretize else 0

            action = (ay, ax, attack, skill_attack, skill_heal)

        #print(f'Action: {action}')

        rl.BeginDrawing()
        rl.BeginMode2D(self.camera)
        rl.ClearBackground([6, 24, 24, 255])

        for y in range(r_min, r_max+1):
            for x in range(c_min, c_max+1):
                if (y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1]):
                    continue

                if pids[y, x] != -1:
                    rl.DrawRectangle(x*ts, y*ts, ts, ts, [0, 0, 255, 128])
 
                tile = grid[y, x]
                if tile == 0:
                    continue
                elif tile == 2:
                    rl.DrawRectangle(x*ts, y*ts, ts, ts, [0, 0, 0, 255])
                    continue
            
                pid = pids[y, x]
                if pid == -1:
                   raise ValueError('Invalid pid')

                entity = entities[pid]

                tx = int(entity.x * ts)
                ty = int(entity.y * ts)

                draw_healthbar(rl, entity, tx, ty, ts)

                #atn = actions[idx]
                source_rect = self.asset_map[tile]
                #dest_rect = (j*ts, i*ts, ts, ts)
                dest_rect = (tx, ty, ts, ts)
                #print(f'tile: {tile}, source_rect: {source_rect}, dest_rect: {dest_rect}')
                rl.DrawTexturePro(self.puffer, source_rect, dest_rect,
                    (0, 0), 0, colors.WHITE)

        # Draw circle at mouse x, y
        rl.DrawCircle(ts*mouse_x + ts//2, ts*mouse_y + ts//2, ts//2, [255, 0, 0, 255])

        rl.EndMode2D()
        rl.EndDrawing()
        return self._cdata_to_numpy(), action

def draw_healthbar(rl, entity, x, y, ts):
    health_bar = entity.health / entity.max_health
    rl.DrawRectangle(x, y - 8, ts, 4, [255, 0, 0, 255])
    rl.DrawRectangle(x, y - 8, int(ts*health_bar), 4, [0, 255, 0, 255])

def test_puffer_performance(timeout):
    import time
    env = PufferGrid()
    actions = np.random.randn(1000, env.num_agents, 2)
    idx = 0
    dones = {1: True}
    start = time.time()
    while time.time() - start < timeout:
        if env.done:
            env.reset()
            dones = {1: False}
        else:
            _, _, dones, _, _ = env.step(actions[idx%1000])

        idx += 1

    sps = env.num_agents * idx // timeout
    print(f'Puffer SPS: {sps}')

if __name__ == '__main__':
    # Run with c profile
    #from cProfile import run
    #run('test_puffer_performance(10)', sort='tottime')
    #exit(0)

    test_puffer_performance(10)
