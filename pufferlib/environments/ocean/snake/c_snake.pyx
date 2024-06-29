# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False
'''Env originally by https://github.com/dnbt777'''

cimport numpy as cnp
from libc.stdlib cimport rand

cdef:
    int EMPTY = 0
    int SNAKE = 1
    int FOOD = 2
    int WALL = 3

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

    def __init__(self, cnp.ndarray grid, cnp.ndarray snake, cnp.ndarray observations,
            snake_lengths, snake_ptr, cnp.ndarray actions,
            cnp.ndarray rewards, int food, int vision):
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

    cdef compute_observations(self):
        for i in range(self.num_snakes):
            head_ptr = self.snake_ptr[i]
            head_r = self.snake[i, head_ptr, 0]
            head_c = self.snake[i, head_ptr, 1]
            self.observations[i] = self.grid[
                head_r - self.vision:head_r + self.vision + 1,
                head_c - self.vision:head_c + self.vision + 1,
            ]


    cdef spawn_snake(self, int snake_id):
        # Delete the snake from the grid
        cdef int head_ptr, head_r, head_c
        while self.snake_lengths[snake_id] > 0:
            head_ptr = self.snake_ptr[snake_id]
            head_r = self.snake[snake_id, head_ptr, 0]
            head_c = self.snake[snake_id, head_ptr, 1]
            self.grid[head_r, head_c] = EMPTY
            self.snake[snake_id, head_ptr, 0] = -1
            self.snake[snake_id, head_ptr, 1] = -1
            self.snake_lengths[snake_id] -= 1

            if head_ptr == 0:
                self.snake_ptr[snake_id] = self.max_snake_length - 1
            else:
                self.snake_ptr[snake_id] -= 1

        # Spawn a new snake
        while True:
            head_r = rand() % (self.height - 1)
            head_c = rand() % (self.width - 1)
            if self.grid[head_r, head_c] == EMPTY:
                break

        self.grid[head_r, head_c] = SNAKE
        self.snake[snake_id, 0, 0] = head_r
        self.snake[snake_id, 0, 1] = head_c
        self.snake_lengths[snake_id] = 1
        self.snake_ptr[snake_id] = 0

    cdef void spawn_food(self):
        cdef int r, c
        while True:
            r = rand() % (self.height - 1)
            c = rand() % (self.width - 1)
            if self.grid[r, c] == EMPTY:
                self.grid[r, c] = FOOD
                return

    cpdef void reset(self):
        self.grid[:self.vision, :] = WALL
        self.grid[:, :self.vision] = WALL
        self.grid[:, self.width-self.vision:] = WALL
        self.grid[self.height-self.vision:, :] = WALL

        for i in range(self.num_snakes):
            self.spawn_snake(i)

        for i in range(self.food):
            self.spawn_food()

        self.compute_observations()

    cpdef float step(self):
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
            if hit and not hit_food:
                self.rewards[i] = -1.0
                self.spawn_snake(i)
                continue

            head_ptr += 1
            if head_ptr >= self.max_snake_length:
                head_ptr = 0
            
            self.snake[i, head_ptr, 0] = next_r
            self.snake[i, head_ptr, 1] = next_c
            self.snake_ptr[i] = head_ptr

            if hit_food:
                self.rewards[i] = 0.1
                self.snake_lengths[i] += 1
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
