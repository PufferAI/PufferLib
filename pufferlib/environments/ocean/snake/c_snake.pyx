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

cdef class CSnake:
    cdef:
        char[:, :] grid
        int[:, :] snake
        int width
        int height
        int max_snake_length
        int head_ptr
        int head_r
        int head_c
        int food_r
        int food_c
        int snake_length

    def __init__(self, cnp.ndarray grid, cnp.ndarray snake):
        self.grid = grid
        self.snake = snake

        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.max_snake_length = self.width * self.height
        self.snake_length = 0

    cpdef void reset(self):
        while self.snake_length > 0:
            self.grid[self.head_r, self.head_c] = EMPTY
            self.snake_length -= 1
            if self.head_ptr == 0:
                self.head_ptr = self.max_snake_length - 1
            else:
                self.head_ptr -= 1

            self.head_r = self.snake[self.head_ptr, 0]
            self.head_c = self.snake[self.head_ptr, 1]

        self.grid[self.food_r, self.food_c] = EMPTY

        self.head_r = int(self.height / 2)
        self.head_c = int(self.width / 2)
        self.head_ptr = 0
        self.snake_length = 1
        self.grid[self.head_r, self.head_c] = 1
        self.place_food()

    cdef void place_food(self):
        cdef int x, y
        while True:
            x = rand() % (self.width - 1)
            y = rand() % (self.height - 1)
            if self.grid[y, x] == EMPTY:
                self.food_r = y
                self.food_c = x
                self.grid[y, x] = FOOD
                return

    cpdef float step(self, int action):
        cdef:
            int dr = 0
            int dc = 0
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

        if action == 0:  # up
            dr = -1
        elif action == 1:  # down
            dr = 1
        elif action == 2:  # left
            dc = -1
        elif action == 3:  # right
            dc = 1

        next_r = self.head_r + dr
        next_c = self.head_c + dc

        hit_wall = (next_r < 0 or next_r >= self.height
            or next_c < 0 or next_c >= self.width)
        hit_self = self.grid[next_r, next_c] == SNAKE
        if hit_wall or hit_self:
            return -1.0

        self.head_ptr += 1
        if self.head_ptr >= self.max_snake_length:
            self.head_ptr = 0

        self.snake[self.head_ptr, 0] = next_r
        self.snake[self.head_ptr, 1] = next_c

        if self.grid[next_r, next_c] == FOOD:
            reward = 0.1
            self.snake_length += 1
            self.place_food()
        else:
            reward = 0.0
            tail_ptr = self.head_ptr - self.snake_length + 1
            if tail_ptr < 0:
                tail_ptr = self.max_snake_length + tail_ptr

            tail_r = self.snake[tail_ptr, 0]
            tail_c = self.snake[tail_ptr, 1]
            self.snake[tail_ptr, 0] = -1
            self.snake[tail_ptr, 1] = -1
            self.grid[tail_r, tail_c] = EMPTY

        self.grid[next_r, next_c] = SNAKE

        dist_to_food = abs(self.food_r - self.head_r) + abs(self.food_c - self.head_c)
        next_dist_food = abs(next_r - self.food_r) + abs(next_c - self.food_c)
        if next_dist_food < dist_to_food:
            reward += 0.01

        return reward
