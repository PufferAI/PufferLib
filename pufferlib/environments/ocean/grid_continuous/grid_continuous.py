from pdb import set_trace as T
import numpy as np

import pettingzoo
import gymnasium

import pufferlib
from pufferlib.environments.ocean.grid_continuous import render
from pufferlib.environments.ocean.grid_continuous.c_grid_continuous import Environment as CEnv

EMPTY = 0
AGENT = 1
WALL = 2

PASS = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4

class PufferGrid(pufferlib.PufferEnv):
    def __init__(self, width=1024, height=1024, num_agents=1024,
            horizon=1024, vision_range=5, agent_speed=1.0,
            discretize=False, reward_fn='introverts', render_mode='rgb_array'):
        super().__init__()
        self.width = width 
        self.height = height
        self.num_agents = num_agents
        self.horizon = horizon
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.obs_size = 2*self.vision_range + 1

        if reward_fn == 'introverts':
            self.reward_fn = self.reward_introverts
        elif reward_fn == 'centralized':
            self.reward_fn = self.reward_centralized
        else:
            raise ValueError(f'reward_fn {reward_fn} must be introverts or centralized')

        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.agent_positions = np.zeros((num_agents, 2), dtype=np.float32)
        self.spawn_position_cands = gen_spawn_positions(width, height)
        self.emulated = None

        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (num_agents, self.obs_size, self.obs_size), dtype=np.uint8),
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
        self.renderer = render.make_renderer(width, height,
            render_mode=render_mode)

        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_size, self.obs_size), dtype=np.uint8)

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

    def reward_introverts(self):
        '''-1 for each nearby agent'''
        raw_rewards = 1 - (self.buf.observations==AGENT).sum(axis=(1,2))
        return np.clip(raw_rewards/10, -1, 0)

    def reward_centralized(self):
        '''Centralized Euclidean distance'''
        pos = self.agent_positions / self.map_size
        sq_diff = (pos[None, :, :] - pos[:, None, :])**2
        return np.sqrt(sq_diff.sum(2)).sum(1) / self.map_size
 
    def render(self):
        return self.renderer.render(self.grid,
            self.agent_positions, self.actions, self.vision_range)

    def reset(self, seed=0):
        if self.cenv is None:
            self.cenv = CEnv(self.grid, self.agent_positions,
                self.spawn_position_cands, self.buf.observations,
                self.width, self.height, self.num_agents, self.horizon,
                self.vision_range, self.agent_speed, self.discretize)

        self.agents = [i+1 for i in range(self.num_agents)]
        self.done = False
        self.tick = 1

        self.grid.fill(0)
        self.episode_rewards.fill(0)
        self.cenv.reset(seed)

        return self.buf.observations, self.infos

    def step(self, actions):
        if self.discretize:
            actions = actions.astype(np.uint32)
        else:
            actions = np.clip(actions, -1, 1).astype(np.float32)

        self.actions = actions
        self.cenv.step(actions)

        self.buf.rewards[:] = self.reward_fn()
        self.episode_rewards[self.tick] = self.buf.rewards
        self.tick += 1

        if self.tick >= self.horizon:
            self.done = True
            self.agents = []
            self.buf.terminals[:] = self.dones
            self.buf.truncations[:] = self.dones
            infos = {'episode_return': self.episode_rewards.sum(1).mean()}
        else:
            self.buf.terminals[:] = self.not_done
            self.buf.truncations[:] = self.not_done
            infos = self.infos

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, infos)

def gen_spawn_positions(width, height):
    '''Generate spawn positions moving outward from the center'''
    y = np.arange(height)
    x = np.arange(width)
    mid_y = height//2 - 0.5
    mid_x = width//2 - 0.5
    positions = np.stack(np.meshgrid(y, x), axis=-1).reshape(-1, 2)
    positions = sorted(positions, key=lambda p: max(abs(p[0]-mid_y), abs(p[1]-mid_x)))
    return np.array(positions, dtype=np.float32)

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
    use_c = False
    test_puffer_performance(10)
