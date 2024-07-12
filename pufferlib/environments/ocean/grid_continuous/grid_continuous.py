from pdb import set_trace as T
import numpy as np
import os

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.environments.ocean.grid_continuous.c_grid_continuous import Environment as CEnv

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

def init_introverts(env):
    pass

def reward_introverts(env):
    '''-1 for each nearby agent'''
    raw_rewards = 1 - (env.buf.observations>=AGENT_1).sum(axis=(1,2))
    return np.clip(raw_rewards/10, -1, 0)

def reward_centralized(env):
    '''Centralized Euclidean distance'''
    pos = env.agent_positions / env.map_size
    sq_diff = (pos[None, :, :] - pos[:, None, :])**2
    return np.sqrt(sq_diff.sum(2)).sum(1) / env.map_size

def init_introverts(env):
    pass

def init_foraging(env, food_prob=0.1):
    rng = np.random.rand(env.height, env.width)
    env.grid[(rng<food_prob) * (env.grid==EMPTY)] = FOOD
    env.grid[(rng>=food_prob) * (env.grid==EMPTY)] = EMPTY

def reward_foraging(env):
    return np.zeros(env.num_agents, dtype=np.float32)

def init_predator_prey(env):
    n = env.num_agents // 2
    env.agent_colors[:n] = AGENT_1
    env.agent_colors[n:] = AGENT_2

def reward_predator_prey(env):
    rewards = np.zeros(env.num_agents, dtype=np.float32)
    n = env.num_agents // 2

    # Predator wants to keep prey in sight
    rewards[:n] = (env.buf.observations[:n] == AGENT_2).sum(axis=(1,2))

    # Prey wants to keep predator out of sight
    rewards[n:] = -(env.buf.observations[n:] == AGENT_1).sum(axis=(1,2))

    return np.clip(rewards/10, -1, 1)

def init_group(env):
    pass

def reward_group(env):
    same_group = env.buf.observations == env.agent_colors[:, None, None]
    same_reward = same_group.sum(axis=(1,2)) - 1
    diff_group = (env.buf.observations>=AGENT_1) * (~same_group)
    diff_reward = diff_group.sum(axis=(1,2))
    rewards = same_reward - diff_reward
    return np.clip(rewards/10, -1, 1)

def init_puffer(env):
    from PIL import Image
    path = os.path.join(*env.__module__.split('.')[:-1], 'pufferlib.png')
    env.puffer = np.array(Image.open(path))
    env.filled = env.puffer[:, :, 3] != 0
    env.red = (env.puffer[:, :, 0] == 255) * env.filled
    env.blue = (env.puffer[:, :, 1] == 255) * env.filled
    env.agent_red = (env.agent_colors == AGENT_1) + (env.agent_colors==AGENT_3)
    env.agent_blue = (env.agent_colors == AGENT_2) + (env.agent_colors==AGENT_4)

def reward_puffer(env):
    agent_position = env.agent_positions.astype(np.int32)
    red = env.red[agent_position[:, 0], agent_position[:, 1]]
    blue = env.blue[agent_position[:, 0], agent_position[:, 1]]
    filled_red = (red * env.agent_red).astype(bool)
    filled_blue = (blue * env.agent_blue).astype(bool)

    r = env.agent_positions[:, 0]/env.height - 0.5
    c = env.agent_positions[:, 1]/env.width - 0.5
    dist = np.sqrt(r**2 + c**2)
   
    return filled_red + filled_blue - 0.01*dist

def init_center(env):
    pass

def reward_center(env):
    r = env.agent_positions[:, 0]/env.height - 0.5
    c = env.agent_positions[:, 1]/env.width - 0.5
    return -0.01*np.sqrt(r**2 + c**2)
 
class PufferGrid(pufferlib.PufferEnv):
    def __init__(self, width=1024, height=1024, num_agents=4096,
            horizon=1024, vision_range=5, agent_speed=1.0,
            discretize=False, food_reward=0.1,
            init_fn=init_center, reward_fn=reward_center,
            #init_fn=init_puffer, reward_fn=reward_puffer,
            #init_fn=init_predator_prey, reward_fn=reward_predator_prey,
            #init_fn=init_group, reward_fn=reward_group,
            expected_lifespan=1000, render_mode='rgb_array'):
        super().__init__()
        self.width = width 
        self.height = height
        self.num_agents = num_agents
        self.horizon = horizon
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.food_reward = food_reward
        self.init_fn = init_fn
        self.reward_fn = reward_fn
        self.expected_lifespan = expected_lifespan

        self.obs_size = 2*self.vision_range + 1
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.agent_positions = np.zeros((num_agents, 2), dtype=np.float32)
        #self.spawn_position_cands = gen_spawn_positions(width, height)
        self.spawn_position_cands = np.random.randint(
            vision_range, (height-vision_range, width-vision_range),
            (10*num_agents, 2)).astype(np.float32)
        self.agent_colors = np.random.randint(3, 7, num_agents, dtype=np.int32)
        self.emulated = None

        self.buf = pufferlib.namespace(
            observations = np.zeros(
                num_agents*self.obs_size*self.obs_size + 3, dtype=np.uint8),
            rewards = np.zeros(num_agents, dtype=np.float32),
            terminals = np.zeros(num_agents, dtype=bool),
            truncations = np.zeros(num_agents, dtype=bool),
            masks = np.ones(num_agents, dtype=bool),
        )
        self.actions = np.zeros(num_agents, dtype=np.uint32)
        self.episode_rewards = np.zeros((horizon, num_agents), dtype=np.float32)

        self.dones = np.ones(num_agents, dtype=bool)
        self.not_done = np.zeros(num_agents, dtype=bool)

        self.render_mode = render_mode
        if render_mode == 'human':
            self.client = RaylibRender(width, height)
     
        self.renderer = make_renderer(width, height,
            render_mode=render_mode)

        self.observation_space = gymnasium.spaces.Box(low=0, high=255,
            shape=(self.obs_size*self.obs_size+3,), dtype=np.uint8)

        if discretize:
            self.action_space = gymnasium.spaces.MultiDiscrete([3, 3])
        else:
            finfo = np.finfo(np.float32)
            self.action_space = gymnasium.spaces.Box(
                low=finfo.min,
                high=finfo.max,
                shape=(2,),
                dtype=np.float32
            )

        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.cenv = None
        self.done = True
        self.infos = {}


    def render(self):
        grid = self.grid
        if self.render_mode == 'rgb_array':
            v = self.vision_range
            frame = COLORS[grid[v:-v-1, v:-v-1]]
            return frame

        return self.renderer.render(self.grid,
            self.agent_positions, self.actions, self.vision_range)

    def _fill_observations(self):
        self.buf.observations[:, -3] = (255*self.agent_positions[:,0]/self.height).astype(np.uint8)
        self.buf.observations[:, -2] = (255*self.agent_positions[:,1]/self.width).astype(np.uint8)
        self.buf.observations[:, -1] = (255*self.buf.rewards).astype(np.uint8)

    def reset(self, seed=0):
        if self.cenv is None:
            obs_view = self.buf.observations[:, 
                :self.obs_size*self.obs_size].reshape(
                self.num_agents, self.obs_size, self.obs_size)
            
            self.cenv = CEnv(self.grid, self.agent_positions,
                self.spawn_position_cands, self.agent_colors, obs_view,
                self.buf.rewards, self.width, self.height, self.num_agents,
                self.horizon, self.vision_range, self.agent_speed,
                self.discretize, self.food_reward, self.expected_lifespan)

        self.agents = [i+1 for i in range(self.num_agents)]
        self.done = False
        self.tick = 1

        self.grid.fill(EMPTY)
        self.init_fn(self)
        self.episode_rewards.fill(0)
        self.cenv.reset(seed)

        self._fill_observations()
        return self.buf.observations, self.infos

    def step(self, actions):
        if self.discretize:
            actions = actions.astype(np.uint32)
        else:
            actions = np.clip(actions, -1, 1).astype(np.float32)

        self.buf.rewards.fill(0)
        self.actions = actions
        self.cenv.step(actions)

        self.buf.rewards[:] += self.reward_fn(self)
        #self.episode_rewards[self.tick] = self.buf.rewards
        self.tick += 1

        '''
        if self.tick >= self.horizon:
            self.done = True
            self.agents = []
            self.buf.terminals[:] = self.dones
            self.buf.truncations[:] = self.dones
            infos = {'episode_return': self.episode_rewards.sum(1).mean()}
        '''
        self.buf.terminals[:] = self.not_done
        self.buf.truncations[:] = self.not_done
        infos = self.infos

        if self.tick % 32 == 0:
            infos['reward'] = self.buf.rewards.mean()

        self._fill_observations()
        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, infos)

def gen_spawn_positions(width, height):
    '''Generate spawn positions moving outward from the center'''
    y = np.arange(height)
    x = np.arange(width)
    mid_y = height//2 - 0.5
    mid_x = width//2 - 0.5
    positions = np.stack(np.meshgrid(y, x), axis=-1).reshape(-1, 2)[::4]
    positions = sorted(positions, key=lambda p: max(abs(p[0]-mid_y), abs(p[1]-mid_x)))
    return np.array(positions, dtype=np.float32)

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
