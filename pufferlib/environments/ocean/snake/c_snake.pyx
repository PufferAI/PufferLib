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
    int FOOD = 1
    int CORPSE = 2
    int WALL = 3

cdef class CMultiSnake:
    cdef:
        int num_envs
        list envs

    def __init__(self, list grids, cnp.ndarray snakes, cnp.ndarray observations,
            snake_lengths, snake_ptrs, snake_colors, cnp.ndarray actions,
            cnp.ndarray rewards, list num_snakes, list num_food, int vision, int max_snake_length,
            list leave_corpse_on_death, list teleport_at_edge):

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
                snake_colors[ptr:end],
                actions[ptr:end],
                rewards[ptr:end],
                num_food[i],
                vision,
                max_snake_length,
                leave_corpse_on_death[i],
                teleport_at_edge[i],
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
        int[:] snake_colors
        unsigned int[:] actions
        float[:] rewards
        int num_snakes
        int width
        int height
        int max_snake_length
        int food
        int vision
        bint leave_corpse_on_death
        bint teleport_at_edge

    def __init__(self, cnp.ndarray grid, cnp.ndarray snake, cnp.ndarray observations,
            snake_lengths, snake_ptr, snake_colors, cnp.ndarray actions, cnp.ndarray rewards,
            int food, int vision, int max_snake_length, bint leave_corpse_on_death, bint teleport_at_edge):
        self.grid = grid
        self.snake = snake
        self.observations = observations
        self.actions = actions
        self.rewards = rewards

        self.num_snakes = snake.shape[0]
        self.snake_lengths = snake_lengths
        self.snake_ptr = snake_ptr
        self.snake_colors = snake_colors

        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.max_snake_length = max_snake_length
        self.food = food
        self.vision = vision
        self.leave_corpse_on_death = leave_corpse_on_death
        self.teleport_at_edge = teleport_at_edge

    cdef void compute_observations(self):
        cdef:
            int i
            int r
            int c
            int head_ptr
            int head_r
            int head_c

            int map_r
            int map_c

        # We do two separate code paths here to save checks and bugs
        if self.teleport_at_edge:
            for i in range(self.num_snakes):
                head_ptr = self.snake_ptr[i]
                head_r = self.snake[i, head_ptr, 0]
                head_c = self.snake[i, head_ptr, 1]
                if (head_r >= self.vision
                        and head_r < self.height - self.vision - 1
                        and head_c >= self.vision
                        and head_c < self.width - self.vision - 1):
                    self.observations[i] = self.grid[
                        head_r - self.vision:head_r + self.vision + 1,
                        head_c - self.vision:head_c + self.vision + 1,
                    ]
                else:
                    for r in range(-self.vision, self.vision+1):
                        for c in range(-self.vision, self.vision+1):
                            # compute wrapped indices
                            map_r = head_r + r
                            map_c = head_c + c

                            if map_r < 0:
                                map_r = self.height + map_r
                            elif map_r >= self.height:
                                map_r = map_r - self.height
                            if map_c < 0:
                                map_c = self.width + map_c
                            elif map_c >= self.width:
                                map_c = map_c - self.width

                            self.observations[i, r+self.vision, c+self.vision] = self.grid[map_r, map_c]
                    
        else:
            for i in range(self.num_snakes):
                head_ptr = self.snake_ptr[i]
                head_r = self.snake[i, head_ptr, 0]
                head_c = self.snake[i, head_ptr, 1]
                try:
                    self.observations[i] = self.grid[
                        head_r - self.vision:head_r + self.vision + 1,
                        head_c - self.vision:head_c + self.vision + 1,
                    ]
                except:
                    print(f'head_r: {head_r}, head_c: {head_c}, head_ptr: {head_ptr}, snake_length: {self.snake_lengths[i]}')
                    exit()

    cdef void spawn_snake(self, int snake_id):
        # Delete the snake from the grid
        cdef int head_ptr, head_r, head_c
        while self.snake_lengths[snake_id] > 0:
            head_ptr = self.snake_ptr[snake_id]
            head_r = self.snake[snake_id, head_ptr, 0]
            head_c = self.snake[snake_id, head_ptr, 1]

            if self.leave_corpse_on_death and self.snake_lengths[snake_id] % 2 == 0:
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

        self.grid[head_r, head_c] = self.snake_colors[snake_id]
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
        if not self.teleport_at_edge:
            # You either teleport from one side to the other and
            # pad vision or you have to border the map
            self.grid[:self.vision, :] = WALL
            self.grid[:, :self.vision] = WALL
            self.grid[:, self.width-self.vision-1:] = WALL
            self.grid[self.height-self.vision-1:, :] = WALL

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
            int tile
            int tail_ptr
            int tail_r
            int tail_c
            int snake_length
            float reward
            bint grow

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

            snake_length = self.snake_lengths[i]
            head_ptr = self.snake_ptr[i]
            head_r = self.snake[i, head_ptr, 0]
            head_c = self.snake[i, head_ptr, 1]
            next_r = head_r + dr
            next_c = head_c + dc

            if self.teleport_at_edge:
                if next_r == -1:
                    next_r = self.height - 1
                elif next_r == self.height:
                    next_r = 0
                if next_c == -1:
                    next_c = self.width - 1
                elif next_c == self.width:
                    next_c = 0

            tile = self.grid[next_r, next_c]

            if tile >= WALL:
                self.rewards[i] = -1.0
                self.spawn_snake(i)
                continue

            head_ptr += 1
            if head_ptr >= self.max_snake_length:
                head_ptr = 0
            
            self.snake[i, head_ptr, 0] = next_r
            self.snake[i, head_ptr, 1] = next_c
            self.snake_ptr[i] = head_ptr

            if tile == FOOD:
                self.rewards[i] = 0.1
                self.spawn_food()
                grow = True
            elif tile == CORPSE:
                self.rewards[i] = 0.1
                grow = True
            else:
                self.rewards[i] = 0.0
                grow = False

            if grow and snake_length < self.max_snake_length:
                self.snake_lengths[i] += 1
            else:
                tail_ptr = head_ptr - snake_length
                if tail_ptr < 0:
                    tail_ptr = self.max_snake_length + tail_ptr

                tail_r = self.snake[i, tail_ptr, 0]
                tail_c = self.snake[i, tail_ptr, 1]
                self.snake[i, tail_ptr, 0] = -1
                self.snake[i, tail_ptr, 1] = -1
                self.grid[tail_r, tail_c] = EMPTY

            self.grid[next_r, next_c] = self.snake_colors[i]

        self.compute_observations()
