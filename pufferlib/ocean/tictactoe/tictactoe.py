'''Pure python version of Squared, a simple single-agent sample environment.
   Use this as a template for your own envs.
'''

# We only use Gymnasium for their spaces API for compatibility with other libraries.
import gymnasium
import numpy as np

import pufferlib

EMPTY = 0
AGENT = 1
ENEMY = 2

# Inherit from PufferEnv
class TicTacToe(pufferlib.PufferEnv):
    # Required keyword arguments: render_mode, buf, seed
    def __init__(self, render_mode='ansi', buf=None, seed=0, num_envs=1):
        # Required attributes below
        size = 3 # tic tac toe is always 3 by 3
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=2,
            shape=(size*size,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(9)
        self.render_mode = render_mode
        self.num_agents = 1 # playing against a random agent

        # Call super after initializing attributes
        super().__init__(buf)

        # Add anything else you want
        self.size = size
        self.num_turns = 0

    # All methods below are required with the signatures shown
    def reset(self, seed=0):
        self.observations[0, :] = EMPTY
        self.num_turns = 0

        np.random.seed(seed)
        turn = np.random.randint(0, 2)

        if turn == 1:
            enemy_move = np.random.randint(0, 9)
            self.observations[0, enemy_move] = ENEMY
            self.num_turns += 1

        # Observations are read from self. Don't create extra copies
        return self.observations, []

    def step(self, actions):
        atn = actions[0]

        # Note that terminals, rewards, etc. are updated in-place
        self.terminals[0] = False
        self.rewards[0] = 0
        info = []

        if self.observations[0, atn] == EMPTY:
            self.observations[0, atn] = AGENT
            self.num_turns += 1
        else:
            self.terminals[0] = True
            self.rewards[0] = -1.0
            return self.observations, self.rewards, self.terminals, self.truncations, [{'reward': -1.0}]

        if self.check_winner(AGENT):
            self.terminals[0] = True
            self.rewards[0] = 1.0
            info = [{'reward': 1.0}]
            self.reset()
        else:
            if self.num_turns == 9:
                self.terminals[0] = True
                self.rewards[0] = 0.0
                info = [{'reward': 0.0}]
                self.reset()
            else:
                while True:
                    enemy_move = np.random.randint(0, 9)
                    if self.observations[0, enemy_move] == EMPTY:
                        self.observations[0, enemy_move] = ENEMY
                        self.num_turns += 1
                        break

                if self.check_winner(ENEMY):
                    self.terminals[0] = True
                    self.rewards[0] = -1.0
                    info = [{'reward': -1.0}]
                    self.reset()
                elif self.num_turns == 9:
                    self.terminals[0] = True
                    self.rewards[0] = 0.0
                    info = [{'reward': 0.0}]
                    self.reset()

        # Return the in-place versions. Don't copy!
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def check_winner(self, player):
        board = self.observations[0].reshape((3, 3))
        for i in range(3):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return True
        if (np.all(board.diagonal() == player) or np.all(np.fliplr(board).diagonal() == player)):
            return True
        return False

    def render(self):
        # Quick ascii rendering. If you want a Python-based renderer,
        # we highly recommend Raylib over PyGame etc. If you use the
        # C-style Python API, it will be very easy to port to C native later.
        chars = []
        grid = self.observations.reshape(self.size, self.size)
        for row in grid:
            for val in row:
                if val == EMPTY:
                    chars.append('. ')
                elif val == AGENT:
                    chars.append('X ')
                elif val == ENEMY:
                    chars.append('O ')
            chars.append('\n')
        return ''.join(chars)

    def close(self):
        pass

if __name__ == '__main__':
    env = TicTacToe()
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 9, (CACHE, 1))

    import time
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[steps % CACHE])
        steps += 1
        print(env.render())

    print('TicTacToe SPS:', int(steps / (time.time() - start)))
