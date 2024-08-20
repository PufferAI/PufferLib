from pdb import set_trace as T
import numpy as np
import os

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.environments.ocean import render
from pufferlib.environments.ocean.moba.c_moba import Environment as CEnv
from pufferlib.environments.ocean.moba.c_moba import entity_dtype, reward_dtype,step_all
from pufferlib.environments.ocean.moba.c_precompute_pathing import precompute_pathing

HUMAN_PLAYER = 4

EMPTY = 0
WALL = 1

PASS = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4

COLORS = np.array([
    [6, 24, 24, 255],     # Empty
    [0, 178, 178, 255],   # Wall
    [255, 165, 0, 255],   # Tower
    [0, 0, 128, 255],   # Radiant Creep
    [128, 0, 0, 255],   # Dire Creep
    [128, 128, 128, 255], # Neutral
    [0, 0, 255, 255],     # Radiant Support
    [0, 0, 255, 255],     # Radiant Assassin
    [0, 0, 255, 255],     # Radiant Burst
    [0, 0, 255, 255],     # Radiant Tank
    [0, 0, 255, 255],     # Radiant Carry
    [255, 0, 0, 255],     # Dire Support
    [255, 0, 0, 255],     # Dire Assassin
    [255, 0, 0, 255],     # Dire Burst
    [255, 0, 0, 255],     # Dire Tank
    [255, 0, 0, 255],     # Dire Carry
], dtype=np.uint8)

PLAYER_OBS_N = 26

class PufferMoba(pufferlib.PufferEnv):
    def __init__(self, num_envs=4, vision_range=5, agent_speed=0.5,
            discretize=True, reward_death=-1.0, reward_xp=0.006,
            reward_distance=0.05, reward_tower=3.0,
            report_interval=32, render_mode='rgb_array'):
        super().__init__()

        self.height = 128
        self.width = 128
        self.num_envs = num_envs
        self.num_agents = 10 * num_envs
        self.num_creeps = 100
        self.num_neutrals = 72
        self.num_towers = 24
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.report_interval = report_interval

        self.reward_death = reward_death
        self.reward_xp = reward_xp
        self.reward_distance = reward_distance
        self.reward_tower = reward_tower

        self.obs_size = 2*self.vision_range + 1
        self.obs_map_bytes = self.obs_size*self.obs_size*4

        # load game map from png
        game_map_path = os.path.join(
            *self.__module__.split('.')[:-1], 'dota_map.png')
        from PIL import Image
        game_map = np.array(Image.open(game_map_path))[:, :, -1]
        game_map = game_map[::2, ::2][1:-1, 1:-1]

        ai_cache_path = os.path.join(
            *self.__module__.split('.')[:-1], 'pathing_cache.npy')

        entity_data_path = os.path.join(
            *self.__module__.split('.')[:-1], 'data.yaml')
        import yaml
        with open(entity_data_path, 'r') as f:
            self.entity_data = yaml.safe_load(f)

        try:
            self.ai_paths = np.load(ai_cache_path)
        except:
            pathing_map = game_map.copy()
            pathing_map[game_map == 0] = 1
            pathing_map[game_map == 255] = 0
            '''
            for k in self.entity_data:
                if 'tower' not in k:
                    continue

                y = int(self.entity_data[k]['y'])
                x = int(self.entity_data[k]['x'])
                pathing_map[y, x] = 1
            '''

            self.ai_paths = np.asarray(precompute_pathing(pathing_map))
            np.save(ai_cache_path, self.ai_paths)

        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.grid[game_map == 0] = WALL
        self.grid[:self.vision_range + 1] = WALL
        self.grid[-self.vision_range - 1:] = WALL
        self.grid[:, :self.vision_range + 1] = WALL
        self.grid[:, -self.vision_range - 1:] = WALL

        self.pids = np.zeros((self.num_envs, self.height, self.width), dtype=np.int32) - 1

        dtype = entity_dtype()
        self.c_entities = np.zeros((self.num_envs, 10 + self.num_creeps +
            self.num_neutrals + self.num_towers), dtype=dtype)
        self.entities = self.c_entities.view(np.recarray)
        self.entities.pid = -1
        self.c_obs_players = np.zeros((self.num_envs, 10, 10), dtype=dtype)
        self.obs_players = self.c_obs_players.view(np.recarray)
        dtype = reward_dtype()
        self.c_rewards = np.zeros((self.num_agents), dtype=dtype)
        self.rewards = self.c_rewards.view(np.recarray)
        self.norm_rewards = np.zeros((self.num_agents), dtype=np.float32)

        self.emulated = None

        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (self.num_agents, self.obs_map_bytes + PLAYER_OBS_N), dtype=np.uint8),
            rewards = np.zeros(self.num_agents, dtype=np.float32),
            terminals = np.zeros(self.num_agents, dtype=bool),
            truncations = np.zeros(self.num_agents, dtype=bool),
            masks = np.ones(self.num_agents, dtype=bool),
        )

        self.render_mode = render_mode
        if render_mode == 'rgb_array':
            self.client = render.RGBArrayRender(colors=COLORS[:, :3])
        elif render_mode == 'raylib':
            self.client = render.GridRender(128, 128,
                screen_width=1024, screen_height=1024, colors=COLORS[:, :3])
        elif render_mode == 'human':
            self.client = RaylibClient(41, 23, COLORS.tolist())
     
        #self.client = render.RGBArrayRender()
        self.observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_map_bytes + PLAYER_OBS_N,), dtype=np.uint8)

        if discretize:
            atn_vec = [7, 7, 3, 2, 2, 2]
            self.action_space = gymnasium.spaces.MultiDiscrete(atn_vec)
            self.actions = np.zeros((self.num_agents, len(atn_vec)), dtype=np.int32)
        else:
            self.actions = np.zeros((self.num_agents, 6), dtype=np.float32)
            finfo = np.finfo(np.float32)
            self.action_space = gymnasium.spaces.Box(
                low=finfo.min,
                high=finfo.max,
                shape=(6,),
                dtype=np.float32
            )

        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.cenv = None
        self.done = True
        self.outcome = 0
        self.infos = {}

    def render(self, upscale=4):
        grid = self.grid
        if self.render_mode in ('rgb_array', 'raylib'):
            #debug_grid = ((grid != EMPTY) * (grid != WALL)).astype(np.uint8)
            return self.client.render(grid)
        elif self.render_mode == 'human':
            frame, self.human_action = self.client.render(
                self.grid, self.pids[0], self.entities[0], self.obs_players[0],
                self.actions[:10], self.discretize)
            return frame
        else:
            raise ValueError(f'Invalid render mode: {self.render_mode}')

    def reset(self, seed=0):
        self.obs_view_map = self.buf.observations[
            :, :self.obs_map_bytes].reshape(
            self.num_envs, 10, self.obs_size, self.obs_size, 4)
        self.obs_view_extra = self.buf.observations[
            :, self.obs_map_bytes:].reshape(
            self.num_envs, 10, PLAYER_OBS_N)
           
        self.agents = [i+1 for i in range(self.num_agents)]
        self.done = False
        self.tick = 1

        '''
        self.grid[
            self.entities.y.astype(np.int32),
            self.entities.x.astype(np.int32)
        ] = AGENT_1
        '''

        grid_copy = self.grid.copy()
        self.sum_rewards = []

        if hasattr(self, 'c_envs'):
            for i in range(self.num_envs):
                self.c_envs[i].reset()

            return self.buf.observations, self.infos

        ptr = end = 0
        self.c_envs = []
        for i in range(self.num_envs):
            end += 10

            # Render env gets true grid
            if i == 0:
                grid = self.grid
            else:
                grid = grid_copy.copy()

            self.c_envs.append(CEnv(grid, self.ai_paths,
                self.pids[i], self.c_entities[i], self.entity_data,
                self.c_obs_players[i], self.obs_view_map[i], self.obs_view_extra[i],
                self.rewards[ptr:end], self.buf.rewards[ptr:end], self.norm_rewards,
                self.actions[ptr:end], 10, self.num_creeps,
                self.num_neutrals, self.num_towers, self.vision_range, self.agent_speed,
                True, self.reward_death, self.reward_xp, self.reward_distance, self.reward_tower))
            self.c_envs[i].reset()
            if i != 0:
                self.c_envs[i].randomize_tower_hp()
            ptr = end

        return self.buf.observations, self.infos

    def step(self, actions):
        #self.actions[:] = actions.astype(np.float32)
        self.actions[:] = actions#.astype(np.float32)

        if self.render_mode == 'human' and self.human_action is not None:
            #print(self.human_action)
            self.actions[HUMAN_PLAYER] = self.human_action
            #print(self.actions[HUMAN_PLAYER])

        step_all(self.c_envs)
        infos = {}


        #print('Reward: ', self.buf.rewards[0])

        '''
        if self.discretize:
            actions = actions.astype(np.uint32)
        else:
            actions = np.clip(actions,
                np.array([-1, -1, 0, 0, 0, 0]),
                np.array([1, 1, 10, 1, 1, 1])
            ).astype(np.float32)
        '''

        #self.outcome = outcome
        #if outcome == 0:
        #    pass
        #elif outcome == 1:
        #    print('Dire Victory')
        #elif outcome == 2:
        #    print('Radiant Victory')

        #self.sum_rewards.append(self.buf.rewards.sum())

        self.tick += 1
        if self.tick % self.report_interval == 0:
            #infos['reward'] = np.mean(self.sum_rewards) / self.num_agents
            infos['reward'] = np.mean(self.buf.rewards)
            levels = self.entities.level
            infos['radiant_level_mean'] = np.mean(levels[:, :5])
            infos['dire_level'] = np.mean(levels[:, 5:10])
            infos['reward_death'] = np.mean(self.rewards.death)
            infos['reward_xp'] = np.mean(self.rewards.xp)
            infos['reward_distance'] = np.mean(self.rewards.distance)
            infos['reward_tower'] = np.mean(self.rewards.tower)
            infos['total_towers_taken'] = np.mean([env.total_towers_taken for env in self.c_envs])
            infos['total_levels_gained'] = np.mean([env.total_levels_gained for env in self.c_envs])
            infos['radiant_victories'] = np.mean([env.radiant_victories for env in self.c_envs])
            infos['dire_victories'] = np.mean([env.dire_victories for env in self.c_envs])
            infos['norm_reward'] = np.mean(self.norm_rewards)

            support = self.entities[:, 0:10:5]
            assassin = self.entities[:, 1:10:5]
            burst = self.entities[:, 2:10:5]
            tank = self.entities[:, 3:10:5]
            carry = self.entities[:, 4:10:5]

            infos['level/support'] = np.mean(support.level)
            infos['level/assassin'] = np.mean(assassin.level)
            infos['level/burst'] = np.mean(burst.level)
            infos['level/tank'] = np.mean(tank.level)
            infos['level/carry'] = np.mean(carry.level)

            infos['deaths/support'] = np.mean(support.deaths)
            infos['deaths/assassin'] = np.mean(assassin.deaths)
            infos['deaths/burst'] = np.mean(burst.deaths)
            infos['deaths/tank'] = np.mean(tank.deaths)
            infos['deaths/carry'] = np.mean(carry.deaths)

            infos['heros_killed/support'] = np.mean(support.heros_killed)
            infos['heros_killed/assassin'] = np.mean(assassin.heros_killed)
            infos['heros_killed/burst'] = np.mean(burst.heros_killed)
            infos['heros_killed/tank'] = np.mean(tank.heros_killed)
            infos['heros_killed/carry'] = np.mean(carry.heros_killed)
            infos['heros_killed/assassin'] = np.mean(assassin.heros_killed)

            infos['damage_dealt/support'] = np.mean(support.damage_dealt)
            infos['damage_dealt/assassin'] = np.mean(assassin.damage_dealt)
            infos['damage_dealt/burst'] = np.mean(burst.damage_dealt)
            infos['damage_dealt/tank'] = np.mean(tank.damage_dealt)
            infos['damage_dealt/carry'] = np.mean(carry.damage_dealt)

            infos['damage_received/support'] = np.mean(support.damage_received)
            infos['damage_received/assassin'] = np.mean(assassin.damage_received)
            infos['damage_received/burst'] = np.mean(burst.damage_received)
            infos['damage_received/tank'] = np.mean(tank.damage_received)
            infos['damage_received/carry'] = np.mean(carry.damage_received)

            infos['healing_dealt/support'] = np.mean(support.healing_dealt)
            infos['healing_dealt/assassin'] = np.mean(assassin.healing_dealt)
            infos['healing_dealt/burst'] = np.mean(burst.healing_dealt)
            infos['healing_dealt/tank'] = np.mean(tank.healing_dealt)
            infos['healing_dealt/carry'] = np.mean(carry.healing_dealt)

            infos['healing_received/support'] = np.mean(support.healing_received)
            infos['healing_received/assassin'] = np.mean(assassin.healing_received)
            infos['healing_received/burst'] = np.mean(burst.healing_received)
            infos['healing_received/tank'] = np.mean(tank.healing_received)
            infos['healing_received/carry'] = np.mean(carry.healing_received)

            infos['usage/support_q'] = np.mean(support.q_uses)
            infos['usage/support_w'] = np.mean(support.w_uses)
            infos['usage/support_e'] = np.mean(support.e_uses)

            infos['usage/assassin_q'] = np.mean(assassin.q_uses)
            infos['usage/assassin_w'] = np.mean(assassin.w_uses)
            infos['usage/assassin_e'] = np.mean(assassin.e_uses)

            infos['usage/burst_q'] = np.mean(burst.q_uses)
            infos['usage/burst_w'] = np.mean(burst.w_uses)
            infos['usage/burst_e'] = np.mean(burst.e_uses)

            infos['usage/tank_q'] = np.mean(tank.q_uses)
            infos['usage/tank_w'] = np.mean(tank.w_uses)
            infos['usage/tank_e'] = np.mean(tank.e_uses)

            infos['usage/carry_q'] = np.mean(carry.q_uses)
            infos['usage/carry_w'] = np.mean(carry.w_uses)
            infos['usage/carry_e'] = np.mean(carry.e_uses)

            #self.sum_rewards = []
            #print('Radiant Lv: ', infos['radiant_level_mean'])
            #print('Dire Lv: ', infos['dire_level_mean'])
            #infos['moba_map'] = self.client.render(self.grid)

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
            2: (512, 0, 128, 128),
            11: (0, 0, 128, 128),
            12: (0, 0, 128, 128),
            13: (0, 0, 128, 128),
            14: (0, 0, 128, 128),
            15: (0, 0, 128, 128),
            6: (128, 0, 128, 128),
            7: (128, 0, 128, 128),
            8: (128, 0, 128, 128),
            9: (128, 0, 128, 128),
            10: (128, 0, 128, 128),
            4: (256, 0, 128, 128),
            3: (384, 0, 128, 128),
            5: (384, 0, 128, 128),
        }

        from raylib import rl, colors
        rl.InitWindow(width*tile_size, height*tile_size,
            "PufferLib Ray Grid".encode())
        rl.SetTargetFPS(10)
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

        ts = self.tile_size
        main_r = entities[HUMAN_PLAYER].y
        main_c = entities[HUMAN_PLAYER].x
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
        ay = int(np.clip((pos.y - ts*self.height//2) / 50, -3, 3)) + 3
        ax = int(np.clip((pos.x - ts*self.width//2) / 50, -3, 3)) + 3

        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        skill = 0
        skill_q = 0
        skill_w = 0
        skill_e = 0
        target_heros = 1
        if rl.IsKeyDown(rl.KEY_Q):
            skill_q = 1
        if rl.IsKeyDown(rl.KEY_W):
            skill_w = 1
        if rl.IsKeyDown(rl.KEY_E):
            skill_e = 1
        if rl.IsKeyDown(rl.KEY_LEFT_SHIFT):
            target_heros = 2

        action = (ay, ax, target_heros, skill_q, skill_w, skill_e)

        rl.BeginDrawing()
        rl.BeginMode2D(self.camera)
        rl.ClearBackground([6, 24, 24, 255])

        for y in range(r_min, r_max+1):
            for x in range(c_min, c_max+1):
                if (y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1]):
                    continue

                tile = grid[y, x]
                if tile == 0:
                    continue
                elif tile == 1:
                    rl.DrawRectangle(x*ts, y*ts, ts, ts, [0, 0, 0, 255])
                    continue

        for y in range(r_min, r_max+1):
            for x in range(c_min, c_max+1):
                if (y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1]):
                    continue

                tile = grid[y, x]
                if tile == EMPTY or tile == WALL:
                    continue
           
                pid = pids[y, x]

                if pid == -1:
                    rl.DrawRectangle(x*ts, y*ts, ts, ts, [255, 0, 0, 255])
                    #raise ValueError(f'Invalid pid {pid} on tile {tile} at {y}, {x}')

                entity = entities[pid]
                if entity.is_hit:
                    rl.DrawRectangle(x*ts, y*ts, ts, ts, [255, 0, 0, 128])
 
                tx = int(entity.x * ts)
                ty = int(entity.y * ts)

                draw_bars(rl, entity, tx, ty-8, ts)

                #atn = actions[idx]
                source_rect = self.asset_map[tile]
                #dest_rect = (j*ts, i*ts, ts, ts)
                dest_rect = (tx, ty, ts, ts)
                #print(f'tile: {tile}, source_rect: {source_rect}, dest_rect: {dest_rect}')
                rl.DrawTexturePro(self.puffer, source_rect, dest_rect,
                    (0, 0), 0, colors.WHITE)

                #if entity.pid == target_pid:
                #    rl.DrawCircle(x*ts + ts//2, y*ts + ts//2, ts//4, [0, 255, 0, 255])

        # Draw circle at mouse x, y
        rl.DrawCircle(ts*mouse_x + ts//2, ts*mouse_y + ts//8, ts//8, [255, 0, 0, 255])

        rl.EndMode2D()

        # Draw HUD
        player = entities[HUMAN_PLAYER]
        hud_y = self.height*ts - 2*ts
        draw_bars(rl, player, 2*ts, hud_y, 10*ts, 24, draw_text=True)

        off_color = [255, 255, 255, 255]
        on_color = [0, 255, 0, 255]

        q_color = on_color if skill_q else off_color
        w_color = on_color if skill_w else off_color
        e_color = on_color if skill_e else off_color

        q_cd = player.q_timer
        w_cd = player.w_timer
        e_cd = player.e_timer

        rl.DrawText(f'Q: {q_cd}'.encode(), 13*ts, hud_y - 20, 40, q_color)
        rl.DrawText(f'W: {w_cd}'.encode(), 17*ts, hud_y - 20, 40, w_color)
        rl.DrawText(f'E: {e_cd}'.encode(), 21*ts, hud_y - 20, 40, e_color)
        rl.DrawText(f'Stun: {player.stun_timer}'.encode(), 25*ts, hud_y - 20, 20, e_color)
        rl.DrawText(f'Move: {player.move_timer}'.encode(), 25*ts, hud_y, 20, e_color)

        rl.EndDrawing()
        return self._cdata_to_numpy(), action

def draw_bars(rl, entity, x, y, width, height=4, draw_text=False):
    health_bar = entity.health / entity.max_health
    mana_bar = entity.mana / entity.max_mana
    if entity.max_health == 0:
        health_bar = 2
    if entity.max_mana == 0:
        mana_bar = 2
    rl.DrawRectangle(x, y, width, height, [255, 0, 0, 255])
    rl.DrawRectangle(x, y, int(width*health_bar), height, [0, 255, 0, 255])

    if entity.entity_type == 0:
        rl.DrawRectangle(x, y - height - 2, width, height, [255, 0, 0, 255])
        rl.DrawRectangle(x, y - height - 2, int(width*mana_bar), height, [0, 255, 255, 255])

    if draw_text:
        health = int(entity.health)
        mana = int(entity.mana)
        max_health = int(entity.max_health)
        max_mana = int(entity.max_mana)
        rl.DrawText(f'Health: {health}/{max_health}'.encode(),
            x+8, y+2, 20, [255, 255, 255, 255])
        rl.DrawText(f'Mana: {mana}/{max_mana}'.encode(),
            x+8, y+2 - height - 2, 20, [255, 255, 255, 255])

        #rl.DrawRectangle(x, y - 2*height - 4, int(width*mana_bar), height, [255, 255, 0, 255])
        rl.DrawText(f'Experience: {entity.xp}'.encode(),
            x+8, y - 2*height - 4, 20, [255, 255, 255, 255])

    elif entity.entity_type == 0:
        rl.DrawText(f'Level: {entity.level}'.encode(),
            x+4, y -2*height - 12, 12, [255, 255, 255, 255])


def test_performance(timeout=20, atn_cache=1024, num_envs=400):
    tick = 0

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', 10*num_envs*tick / (time.time() - start))

if __name__ == '__main__':
    # Run with c profile
    from cProfile import run
    num_envs = 100
    env = PufferMoba(num_envs=num_envs, report_interval=10000000)
    env.reset()
    actions = np.random.randint(0, 9, (1024, 10*num_envs))
    test_performance(20, 1024, num_envs)
    exit(0)

    run('test_performance(20)', 'stats.profile')
    import pstats
    from pstats import SortKey
    p = pstats.Stats('stats.profile')
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)

    #test_performance(10)
