# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
'''Env originally by https://github.com/dnbt777'''


import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand

cdef:
    int EMPTY = 0
    int SNAKE = 1
    int FOOD = 2
    int CORPSE = 3
    int WALL = 4

cdef class CMultiSnake:
    cdef:
        char[:, :, :] grids
        char[:, :, :] observations
        char[:, :, :] snakes
        char[:] snake_ptrs
        int[:] snake_lengths
        unsigned int[:] actions
        float[:] rewards
        int num_envs
        list envs

    def __init__(self, list grids, cnp.ndarray snakes, cnp.ndarray observations,
            snake_lengths, snake_ptrs, cnp.ndarray actions, cnp.ndarray rewards,
            list num_snakes, list num_food, int vision, list leave_corpse_on_death):

        cdef int ptr = 0
        cdef int end = 0
        self.num_envs = len(grids)
        self.envs = []
        for i in range(self.num_envs):
            end += num_snakes[i]
            self.envs.append(CSnake(
                grids[i],
                snakes[ptr:end],
                observations[ptr:end],
                snake_lengths[ptr:end],
                snake_ptrs[ptr:end],
                actions[ptr:end],
                rewards[ptr:end],
                num_food[i],
                vision,
                leave_corpse_on_death[i],
            ))
            ptr = end

    cpdef void reset(self):
        cdef:
            CSnake env

        for i in range(self.num_envs):
            env = self.envs[i]
            env.reset()

    cpdef void step(self):
        cdef:
            CSnake env

        for i in range(self.num_envs):
            env = self.envs[i]
            env.step()

cdef class CSnake:
    cdef:
        char[:, :] grid
        char[:, :, :] observations
        int[:, :, :] snake
        int[:] snake_ptr
        int[:] snake_lengths
        unsigned int[:] actions
        float[:] rewards
        int num_snakes
        int width
        int height
        int max_snake_length
        int food
        int vision
        bint leave_corpse_on_death

    def __init__(self, cnp.ndarray grid, cnp.ndarray snake, cnp.ndarray observations,
            snake_lengths, snake_ptr, cnp.ndarray actions, cnp.ndarray rewards,
            int food, int vision, bint leave_corpse_on_death):
        self.grid = grid
        self.snake = snake
        self.observations = observations
        self.actions = actions
        self.rewards = rewards

        self.num_snakes = snake.shape[0]
        self.snake_lengths = snake_lengths
        self.snake_ptr = snake_ptr

        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.max_snake_length = self.width * self.height
        self.food = food
        self.vision = vision
        self.leave_corpse_on_death = leave_corpse_on_death

    cdef void compute_observations(self):
        cdef:
            int i
            int head_ptr
            int head_r
            int head_c

        for i in range(self.num_snakes):
            head_ptr = self.snake_ptr[i]
            head_r = self.snake[i, head_ptr, 0]
            head_c = self.snake[i, head_ptr, 1]
            self.observations[i] = self.grid[
                head_r - self.vision:head_r + self.vision + 1,
                head_c - self.vision:head_c + self.vision + 1,
            ]

    cdef void spawn_snake(self, int snake_id):
        # Delete the snake from the grid
        cdef int head_ptr, head_r, head_c
        while self.snake_lengths[snake_id] > 0:
            head_ptr = self.snake_ptr[snake_id]
            head_r = self.snake[snake_id, head_ptr, 0]
            head_c = self.snake[snake_id, head_ptr, 1]

            if self.leave_corpse_on_death:
                self.grid[head_r, head_c] = CORPSE
            else:
                self.grid[head_r, head_c] = EMPTY

            self.snake[snake_id, head_ptr, 0] = -1
            self.snake[snake_id, head_ptr, 1] = -1
            self.snake_lengths[snake_id] -= 1

            if head_ptr == 0:
                self.snake_ptr[snake_id] = self.max_snake_length - 1
            else:
                self.snake_ptr[snake_id] -= 1

        # Spawn a new snake
        cdef int tile
        while True:
            head_r = rand() % (self.height - 1)
            head_c = rand() % (self.width - 1)
            tile = self.grid[head_r, head_c]
            if tile == EMPTY or tile == CORPSE:
                break

        self.grid[head_r, head_c] = SNAKE
        self.snake[snake_id, 0, 0] = head_r
        self.snake[snake_id, 0, 1] = head_c
        self.snake_lengths[snake_id] = 1
        self.snake_ptr[snake_id] = 0

    cdef void spawn_food(self):
        cdef int r, c, tile
        while True:
            r = rand() % (self.height - 1)
            c = rand() % (self.width - 1)
            tile = self.grid[r, c]
            if tile == EMPTY or tile == CORPSE:
                self.grid[r, c] = FOOD
                return

    cdef void reset(self):
        self.grid[:self.vision+1, :] = WALL
        self.grid[:, :self.vision+1] = WALL
        self.grid[:, self.width-self.vision-2:] = WALL
        self.grid[self.height-self.vision-2:, :] = WALL

        for i in range(self.num_snakes):
            self.spawn_snake(i)

        for i in range(self.food):
            self.spawn_food()

        self.compute_observations()

    cdef float step(self):
        cdef:
            int atn
            int dr
            int dc
            int head_ptr
            int head_r
            int head_c
            int next_r
            int next_c
            int hit_wall
            int hit_self
            int tail_ptr
            int tail_r
            int tail_c
            int dist_to_food
            int next_dist_food
            float reward
            bint hit
            bint hit_food
            bint hit_corpse

        for i in range(self.num_snakes):
            atn = self.actions[i]
            dr = 0
            dc = 0
            if atn == 0:  # up
                dr = -1
            elif atn == 1:  # down
                dr = 1
            elif atn == 2:  # left
                dc = -1
            elif atn == 3:  # right
                dc = 1

            head_ptr = self.snake_ptr[i]
            head_r = self.snake[i, head_ptr, 0]
            head_c = self.snake[i, head_ptr, 1]
            next_r = head_r + dr
            next_c = head_c + dc

            hit = self.grid[next_r, next_c] != EMPTY
            hit_food = self.grid[next_r, next_c] == FOOD
            hit_corpse = self.grid[next_r, next_c] == CORPSE
            if hit and not hit_food and not hit_corpse:
                self.rewards[i] = -1.0
                self.spawn_snake(i)
                continue

            head_ptr += 1
            if head_ptr >= self.max_snake_length:
                head_ptr = 0
            
            self.snake[i, head_ptr, 0] = next_r
            self.snake[i, head_ptr, 1] = next_c
            self.snake_ptr[i] = head_ptr

            if hit_food or hit_corpse:
                self.rewards[i] = 0.1
                self.snake_lengths[i] += 1
                if hit_food:
                    self.spawn_food()
            else:
                self.rewards[i] = 0.0
                tail_ptr = head_ptr - self.snake_lengths[i]
                if tail_ptr < 0:
                    tail_ptr = self.max_snake_length + tail_ptr

                tail_r = self.snake[i, tail_ptr, 0]
                tail_c = self.snake[i, tail_ptr, 1]
                self.snake[i, tail_ptr, 0] = -1
                self.snake[i, tail_ptr, 1] = -1
                self.grid[tail_r, tail_c] = EMPTY

            self.grid[next_r, next_c] = SNAKE

        self.compute_observations()
