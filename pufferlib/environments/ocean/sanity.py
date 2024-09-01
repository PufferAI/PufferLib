import gymnasium
import pettingzoo
import numpy as np
import random
import time


class Bandit(gymnasium.Env):
    '''Pufferlib Bandit environment

    Simulates a classic multiarmed bandit problem.

    Observation space: Box(0, 1, (1,)). The observation is always 1.
    Action space: Discrete(num_actions). Which arm to pull.
    Args:
        num_actions: The number of bandit arms
        reward_scale: The scale of the reward
        reward_noise: The standard deviation of the reward signal
        hard_fixed_seed: All instances of the environment should share the same seed.
    '''
    def __init__(self, num_actions=4, reward_scale=1,
            reward_noise=0, hard_fixed_seed=42):
        self.num_actions = num_actions
        self.reward_scale = reward_scale
        self.reward_noise = reward_noise
        self.hard_fixed_seed = hard_fixed_seed
        self.observation=np.ones(1, dtype=np.float32)
        self.observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(1,))
        self.action_space=gymnasium.spaces.Discrete(num_actions)
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        # Bandit problem requires a single fixed seed
        # for all environments
        seed = self.hard_fixed_seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.solution_idx = np.random.randint(0, self.num_actions)

        return self.observation, {}

    def step(self, action):
        assert action == int(action) and action >= 0 and action < self.num_actions

        correct = False
        reward = 0
        if action == self.solution_idx:
            correct = True
            reward = 1

        reward_noise = 0
        if self.reward_noise != 0:
            reward_noise = np.random.randn() * self.reward_scale

        # Couples reward noise to scale
        reward = (reward + reward_noise) * self.reward_scale

        return self.observation, reward, True, False, {'score': correct}

class Memory(gymnasium.Env):
    '''Pufferlib Memory environment

    Repeat the observed sequence after a delay. It is randomly generated upon every reset. This is a test of memory length and capacity. It starts requiring credit assignment if you make the sequence too long.

    The sequence is presented one digit at a time, followed by a string of 0. The agent should output 0s for the first mem_length + mem_delay steps, then output the sequence.

    Observation space: Box(0, 1, (1,)). The current digit.
    Action space: Discrete(2). Your guess for the next digit.

    Args:
        mem_length: The length of the sequence
        mem_delay: The number of 0s between the sequence and the agent's response
    '''
    def __init__(self, mem_length=1, mem_delay=0):
        self.mem_length = mem_length
        self.mem_delay = mem_delay
        self.horizon = 2 * mem_length + mem_delay
        self.observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(1,))
        self.action_space=gymnasium.spaces.Discrete(2)
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.solution = np.random.randint(0, 2, size=self.horizon).astype(np.float32)
        self.solution[-(self.mem_length + self.mem_delay):] = -1
        self.submission = np.zeros(self.horizon) - 1
        self.tick = 1

        return self.solution[0], {}

    def step(self, action):
        assert self.tick < self.horizon
        assert action in (0, 1)

        ob = reward = 0.0

        if self.tick < self.mem_length:
            ob = self.solution[self.tick]
            reward = float(action == 0)

        if self.tick >= self.mem_length + self.mem_delay:
            idx = self.tick - self.mem_length - self.mem_delay
            sol = self.solution[idx]
            reward = float(action == sol)
            self.submission[self.tick] = action

        self.tick += 1
        terminal = self.tick == self.horizon

        info = {}
        if terminal:
            info['score'] = np.all(
                self.solution[:self.mem_length] == self.submission[-self.mem_length:])

        return ob, reward, terminal, False, info

    def render(self):
        def _render(val):
            if val == 1:
                c = 94
            elif val == 0:
                c = 91
            else:
                c = 90
            return f'\033[{c}m██\033[0m'

        chars = []
        for val in self.solution:
            c = _render(val)
            chars.append(c)
        chars.append(' Solution\n')

        for val in self.submission:
            c = _render(val)
            chars.append(c)
        chars.append(' Prediction\n')

        return ''.join(chars)


class Multiagent(pettingzoo.ParallelEnv):
    '''Pufferlib Multiagent environment

    Agent 1 must pick action 0 and Agent 2 must pick action 1

    Observation space: Box(0, 1, (1,)). 0 for Agent 1 and 1 for Agent 2
    Action space: Discrete(2). Which action to take.
    '''
    def __init__(self):
        self.observation = {
            1: np.zeros(1, dtype=np.float32),
            2: np.ones(1, dtype=np.float32),
        }
        self.terminal = {
            1: True,
            2: True,
        }
        self.truncated = {
            1: False,
            2: False,
        }
        self.possible_agents=[1, 2]
        self.agents=[1, 2]
        self.render_mode = 'ansi'

    def observation_space(self, agent):
        return gymnasium.spaces.Box(
            low=0, high=1, shape=(1,))

    def action_space(self, agent):
        return gymnasium.spaces.Discrete(2)

    def reset(self, seed=None):
        # Reallocating is faster than zeroing
        self.view=np.zeros((2, 5), dtype=np.float32)
        return self.observation, {}

    def step(self, action):
        reward = {}
        assert 1 in action and action[1] in (0, 1)
        if action[1] == 0:
            self.view[0, 2] = 1
            reward[1] = 1
        else:
            self.view[0, 0] = 1
            reward[1] = 0

        assert 2 in action and action[2] in (0, 1)
        if action[2] == 1:
            self.view[1, 2] = 1
            reward[2] = 1
        else:
            self.view[1, 4] = 1
            reward[2] = 0

        info = {
            1: {'score': reward[1]},
            2: {'score': reward[2]},
        }
        return self.observation, reward, self.terminal, self.truncated, info

    def render(self):
        def _render(val):
            if val == 1:
                c = 94
            elif val == 0:
                c = 90
            else:
                c = 90
            return f'\033[{c}m██\033[0m'

        chars = []
        for row in self.view:
            for val in row:
                c = _render(val)
                chars.append(c)
            chars.append('\n')
        return ''.join(chars)

class Password(gymnasium.Env):
    '''Pufferlib Password environment

    Guess the password, which is a static binary string. Your policy has to
    not determinize before it happens to get the reward, and it also has to
    latch onto the reward within a few instances of getting it. 

    Observation space: Box(0, 1, (password_length,)). A binary vector containing your guesses so far, so that the environment will be solvable without memory.
    Action space: Discrete(2). Your guess for the next digit.

    Args:
        password_length: The number of binary digits in the password.
        hard_fixed_seed: A fixed seed for the environment. It should be the same for all instances. This environment does not make sense when randomly generated.
    '''
 
    def __init__(self, password_length=5, hard_fixed_seed=42):
        self.password_length = password_length
        self.hard_fixed_seed = hard_fixed_seed
        self.observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(password_length,))
        self.action_space=gymnasium.spaces.Discrete(2)
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        # Bandit problem requires a single fixed seed
        # for all environments
        seed = self.hard_fixed_seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.observation = np.zeros(self.password_length, dtype=np.float32) - 1
        self.solution = np.random.randint(
            0, 2, size=self.password_length).astype(np.float32)
        self.tick = 0

        return self.observation, {}

    def step(self, action):
        assert self.tick < self.password_length
        assert action in (0, 1)

        self.observation[self.tick] = action
        self.tick += 1

        reward = 0
        terminal = self.tick == self.password_length
        info = {}

        if terminal:
            reward = float(np.all(self.observation == self.solution))
            info['score'] = reward

        return self.observation, reward, terminal, False, info

    def render(self):
        def _render(val):
            if val == 1:
                c = 94
            elif val == 0:
                c = 91
            else:
                c = 90
            return f'\033[{c}m██\033[0m'

        chars = []
        for val in self.solution:
            c = _render(val)
            chars.append(c)
        chars.append(' Solution\n')

        for val in self.observation:
            c = _render(val)
            chars.append(c)
        chars.append(' Prediction\n')

        return ''.join(chars)

class Performance(gymnasium.Env):
    def __init__(self, delay_mean=0, delay_std=0, bandwidth=1):
        np.random.seed(time.time_ns() % 2**32)

        self.observation_space = gymnasium.spaces.Box(
            low=-2**20, high=2**20,
            shape=(bandwidth,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation = self.observation_space.sample()
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        return self.observation, {}

    def step(self, action):
        start = time.process_time()
        idx = 0
        target_time = self.delay_mean + self.delay_std*np.random.randn()
        while time.process_time() - start < target_time:
            idx += 1

        return self.observation, 0, False, False, {}

class PerformanceEmpiric(gymnasium.Env):
    def __init__(self, count_n=0, count_std=0, bandwidth=1):
        np.random.seed(time.time_ns() % 2**32)

        self.observation_space = gymnasium.spaces.Box(
            low=-2**20, high=2**20,
            shape=(bandwidth,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation = self.observation_space.sample()
        self.count_n = count_n
        self.count_std = count_std
        self.bandwidth = bandwidth
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        return self.observation, {}

    def step(self, action):
        idx = 0
        target = self.count_n  +  self.count_std * np.random.randn()
        while idx < target:
            idx += 1

        return self.observation, 0, False, False, {}

class Spaces(gymnasium.Env):
    '''Pufferlib Spaces environment

    A simple environment with hierarchical observation and action spaces

    The image action should be 1 if the sum of the image is positive, 0 otherwise
    The flat action should be 1 if the sum of the flat obs is positive, 0 otherwise

    0.5 reward is given for each correct action

    Does not provide rendering
    '''
    def __init__(self):
        self.observation_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Box(
                low=0, high=1, shape=(5, 5), dtype=np.float32),
            'flat': gymnasium.spaces.Box(
                low=0, high=1, shape=(5,), dtype=np.int8),
        })
        self.action_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Discrete(2),
            'flat': gymnasium.spaces.Discrete(2),
        })
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        self.observation = {
            'image': np.random.randn(5, 5).astype(np.float32),
            'flat': np.random.randint(-1, 2, (5,), dtype=np.int8),
        }
        self.image_sign = np.sum(self.observation['image']) > 0
        self.flat_sign = np.sum(self.observation['flat']) > 0

        return self.observation, {}

    def step(self, action):
        assert isinstance(action, dict)
        assert 'image' in action and action['image'] in (0, 1)
        assert 'flat' in action and action['flat'] in (0, 1)

        reward = 0
        if self.image_sign == action['image']:
            reward += 0.5

        if self.flat_sign == action['flat']:
            reward += 0.5

        info = dict(score=reward)
        return self.observation, reward, True, False, info

class Squared(gymnasium.Env):
    '''Pufferlib Squared environment

    Agent starts at the center of a square grid.
    Targets are placed on the perimeter of the grid.
    Reward is 1 minus the L-inf distance to the closest target.
    This means that reward varies from -1 to 1.
    Reward is not given for targets that have already been hit.

    Observation space: Box(-1, 1, (grid_size, grid_size)). The map.
    Action space: Discrete(8). Which direction to move.

    Args:
        distance_to_target: The distance from the center to the closest target.
        num_targets: The number of targets to randomly generate.
 
    '''

    MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)]

    def __init__(self,
        distance_to_target=1,
        num_targets=-1,
        ):
        grid_size = 2 * distance_to_target + 1
        if num_targets == -1:
            num_targets = 4 * distance_to_target

        self.distance_to_target = distance_to_target
        self.possible_targets = self._all_possible_targets(grid_size)
        self.num_targets = num_targets
        self.grid_size = grid_size
        self.max_ticks = num_targets * distance_to_target
        self.observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(grid_size, grid_size))
        self.action_space = gymnasium.spaces.Discrete(8)
        self.render_mode = 'ansi'

    def _all_possible_targets(self, grid_size):
        return [(x, y) for x in range(grid_size) for y in range(grid_size)
                if x == 0 or y == 0 or x == grid_size - 1 or y == grid_size - 1]

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Allocating a new grid is faster than resetting an old one
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.grid[self.distance_to_target, self.distance_to_target] = -1
        self.agent_pos = (self.distance_to_target, self.distance_to_target)
        self.tick = 0

        self.targets = random.sample(self.possible_targets, self.num_targets)
        for x, y in self.targets:
            self.grid[x, y] = 1

        return self.grid, {}

    def step(self, action):
        x, y = self.agent_pos
        self.grid[x, y] = 0

        dx, dy = Squared.MOVES[action]
        x += dx
        y += dy

        min_dist = min([max(abs(x-tx), abs(y-ty)) for tx, ty in self.targets])
        # This reward function will return 0.46 average reward for an unsuccessful
        # episode with distance_to_target=4 and num_targets=1 (0.5 for solve)
        # It looks reasonable but is not very discriminative
        reward = 1 - min_dist / self.distance_to_target

        # This reward function will return 1 when the agent moves in the right direction
        # (plus an adjustment for the 0 reset reward) to average 1 for success
        # It is not much better than the previous one.
        #reward = state.distance_to_target - min_dist - state.tick + 1/state.max_ticks

        # This function will return 0, 0.2, 0.4, ... 1 for successful episodes (n=5)
        # And will drop rewards to 0 or less as soon as an error is made
        # Somewhat smoother but actually worse than the previous ones
        # reward = (state.distance_to_target - min_dist - state.tick) / (state.max_ticks - state.tick)


        # This one nicely tracks the task completed metric but does not optimize well
        #if state.distance_to_target - min_dist - state.tick  == 1:
        #    reward = 1
        #else:
        #    reward = -state.tick

        if (x, y) in self.targets:
            self.targets.remove((x, y))
            #state.grid[x, y] = 0

        dist_from_origin = max(abs(x-self.distance_to_target), abs(y-self.distance_to_target))
        if dist_from_origin >= self.distance_to_target:
            self.agent_pos = self.distance_to_target, self.distance_to_target
        else:
            self.agent_pos = x, y
        
        self.grid[self.agent_pos] = -1
        self.tick += 1

        done = self.tick >= self.max_ticks
        score = (self.num_targets - len(self.targets)) / self.num_targets
        info = {'score': score} if done else {}

        return self.grid, reward, done, False, info

    def render(self):
        chars = []
        for row in self.grid:
            for val in row:
                if val == 1:
                    color = 94
                elif val == -1:
                    color = 91
                else:
                    color = 90
                chars.append(f'\033[{color}m██\033[0m')
            chars.append('\n')
        return ''.join(chars)

class Stochastic(gymnasium.Env):
    '''Pufferlib Stochastic environment

    The optimal policy is to play action 0 < p % of the time and action 1 < (1 - p) %
    This is a test of whether your algorithm can learn a nontrivial stochastic policy.
    Do not use a policy with memory, as that will trivialize the problem.

    Observation space: Box(0, 1, (1,)). The observation is always 0.
    Action space: Discrete(2). Select action 0 or action 1.

    Args:
        p: The optimal probability for action 0
        horizon: How often the environment should reset
    '''
    def __init__(self, p=0.75, horizon=1000):
        self.p = p
        self.horizon = horizon
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(1,))
        self.action_space = gymnasium.spaces.Discrete(2)
        self.render_mode = 'ansi'

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.tick = 0
        self.count = 0
        self.action = 0

        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        assert self.tick < self.horizon
        assert action in (0, 1)

        self.tick += 1
        self.count += action == 0
        self.action = action

        terminal = self.tick == self.horizon
        atn0_frac = self.count / self.tick
        proximity_to_p = 1 - (self.p - atn0_frac)**2

        reward = proximity_to_p if (
            (action == 0 and atn0_frac < self.p) or
            (action == 1 and atn0_frac >= self.p)) else 0

        info = {}
        if terminal:
            info['score'] = proximity_to_p

        return np.zeros(1, dtype=np.float32), reward, terminal, False, info

    def render(self):
        def _render(val):
            if val == 1:
                c = 94
            elif val == 0:
                c = 91
            else:
                c = 90
            return f'\033[{c}m██\033[0m'
        chars = []
        if self.tick == 0:
            solution = 0
        else:
            solution = 0 if self.count / self.tick < self.p else 1
        chars.append(_render(solution))
        chars.append(' Solution\n')

        chars.append(_render(self.action))
        chars.append(' Prediction\n')

        return ''.join(chars)

class Continuous(gymnasium.Env):
    def __init__(self, discretize=False):
        self.observation_space=gymnasium.spaces.Box(
            low=-1, high=1, shape=(6,))
        self.discretize = discretize
        if discretize:
            self.action_space=gymnasium.spaces.Discrete(4)
        else:
            self.action_space=gymnasium.spaces.Box(
                low=-1, high=1, shape=(2,))

        self.render_mode = 'human'
        self.client = None

    def reset(self, seed=None, options=None):
        # pos_x, pos_y, vel_x, vel_y, target_x, target_y
        self.state = 2*np.random.rand(6)-1
        self.state[2:4] = 0
        self.tick = 0

        return self.state, {}

    def step(self, action):
        if self.discretize:
            accel_x, accel_y = 0, 0
            if action == 0:
                accel_x = -0.1
            elif action == 1:
                accel_x = 0.1
            elif action == 2:
                accel_y = -0.1
            elif action == 3:
                accel_y = 0.1
        else:
            accel_x, accel_y = 0.1*action

        self.state[2] += accel_x
        self.state[3] += accel_y
        self.state[0] += self.state[2]
        self.state[1] += self.state[3]

        pos_x, pos_y, vel_x, vel_y, target_x, target_y = self.state

        if pos_x < -1 or pos_x > 1 or pos_y < -1 or pos_y > 1:
            return self.state, -1, True, False, {'score': 0}

        dist = np.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2)
        reward = 0.02 * (1 - dist)

        self.tick += 1
        done = dist < 0.1
        truncated = self.tick >= 100

        # TODO: GAE implementation making agent not hit target
        # without a big reward here
        info = {}
        if done:
            reward = 5.0
            info = {'score': 1}
        elif truncated:
            reward = 0.0
            info = {'score': 0}

        return self.state, reward, done, truncated, info

    def render(self):
        if self.client is None:
            self.client = RaylibClient()

        pos_x, pos_y, vel_x, vel_y, target_x, target_y = self.state
        frame, atn = self.client.render(pos_x, pos_y, target_x, target_y)
        return frame

class RaylibClient:
    def __init__(self, width=1080, height=720, size=20):
        self.width = width
        self.height = height
        self.size = size

        from raylib import rl
        rl.InitWindow(width, height,
            "PufferLib Simple Continuous".encode())
        rl.SetTargetFPS(10)
        self.rl = rl

        from cffi import FFI
        self.ffi = FFI()

    def _cdata_to_numpy(self):
        image = self.rl.LoadImageFromScreen()
        width, height, channels = image.width, image.height, 4
        cdata = self.ffi.buffer(image.data, width*height*channels)
        return np.frombuffer(cdata, dtype=np.uint8
            ).reshape((height, width, channels))[:, :, :3]

    def render(self, pos_x, pos_y, target_x, target_y):
        rl = self.rl
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
        rl.ClearBackground([6, 24, 24, 255])

        pos_x = int((0.5+pos_x/2) * self.width)
        pos_y = int((0.5+pos_y/2) * self.height)
        target_x = int((0.5+target_x/2) * self.width)
        target_y = int((0.5+target_y/2) * self.height)

        rl.DrawCircle(pos_x, pos_y, self.size, [255, 0, 0, 255])
        rl.DrawCircle(target_x, target_y, self.size, [0, 0, 255, 255])

        rl.EndDrawing()
        return self._cdata_to_numpy(), action
