from pdb import set_trace as T
import numpy as np
import os

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.environments.ocean import render
#from pufferlib.environments.ocean.moba.c_moba import Environment as CEnv
#from pufferlib.environments.ocean.moba.c_moba import entity_dtype, reward_dtype,step_all
from pufferlib.environments.ocean.moba.cy_moba import Environment as CEnv
from pufferlib.environments.ocean.moba.cy_moba import entity_dtype, reward_dtype, step_all
from pufferlib.environments.ocean.moba.c_precompute_pathing import precompute_pathing

HUMAN_PLAYER = 1

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
    def __init__(self, num_envs=4, vision_range=5, agent_speed=1.0,
            discretize=True, reward_death=-1.0, reward_xp=0.006,
            reward_distance=0.05, reward_tower=3.0,
            report_interval=32, render_mode='human'):
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
        self.render_mode = render_mode

        # load game map from png
        import sys
        sys.path.append(os.path.join(*self.__module__.split('.')[:-1]))

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

        self.ai_path_buffer = np.zeros((3*8*self.height*self.width), dtype=np.int32)

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

        '''
        self.render_mode = render_mode
        if render_mode == 'rgb_array':
            self.client = render.RGBArrayRender(colors=COLORS[:, :3])
        elif render_mode == 'raylib':
            self.client = render.GridRender(128, 128,
                screen_width=1024, screen_height=1024, colors=COLORS[:, :3])
        elif render_mode == 'human':
            self.client = RaylibClient(41, 23, COLORS.tolist())
        '''
     
        #self.client = render.RGBArrayRender()
        self.observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_map_bytes + PLAYER_OBS_N,), dtype=np.uint8)

        atn_vec = [7, 7, 3, 2, 2, 2]
        self.actions = np.zeros((self.num_agents, len(atn_vec)), dtype=np.int32)

        if discretize:
            self.action_space = gymnasium.spaces.MultiDiscrete(atn_vec)
        else:
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
        for tick in range(12):
            self.c_envs[0].render(tick)

        '''
        if self.render_mode in ('rgb_array', 'raylib'):
            #debug_grid = ((grid != EMPTY) * (grid != WALL)).astype(np.uint8)
            return self.client.render(grid)
        elif self.render_mode == 'human':
            self.c_envs[0].render()
            #frame, self.human_action = self.client.render(
            #    self.grid, self.pids[0], self.entities[0], self.obs_players[0],
            #    self.actions[:10], self.discretize, 12)
            #return frame
        else:
            raise ValueError(f'Invalid render mode: {self.render_mode}')
        '''

    def reset(self, seed=0):
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

            self.c_envs.append(CEnv(grid, self.ai_paths, self.ai_path_buffer,
                self.pids[i], self.c_entities[i], self.entity_data,
                self.c_obs_players[i], self.buf.observations[ptr:end],
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
        self.actions[:] = actions
        self.actions[:, 0] = 100*(self.actions[:, 0] - 3)
        self.actions[:, 1] = 100*(self.actions[:, 1] - 3)
        '''
        if self.render_mode == 'human' and self.human_action is not None:
            #print(self.human_action)
            self.actions[HUMAN_PLAYER] = self.human_action
            #print(self.actions[HUMAN_PLAYER])
        '''

        for i in range(self.num_envs):
            self.c_envs[i].step()

        #step_all(self.c_envs)
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
