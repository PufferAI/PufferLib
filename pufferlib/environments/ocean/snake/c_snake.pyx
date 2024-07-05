# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

from cpython.list cimport PyList_GET_ITEM
cimport numpy as cnp
from libc.stdlib cimport rand

cdef:
    int EMPTY = 0
    int FOOD = 1
    int CORPSE = 2
    int WALL = 3

def step_all(list envs):
    cdef int n = len(envs)
    for i in range(n):
        (<CSnake>PyList_GET_ITEM(envs, i)).step()

cdef class CSnake:
    cdef:
        char[:, :] grid
        char[:, :, :] observations
        int[:, :, :] snake
        int[:] snake_lengths
        int[:] snake_ptr
        int[:] snake_lifetimes
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
        float reward_food
        float reward_corpse
        float reward_death

    def __init__(self, cnp.ndarray grid, cnp.ndarray snake,
            cnp.ndarray observations, cnp.ndarray snake_lengths,
            cnp.ndarray snake_ptr, cnp.ndarray snake_lifetimes,
            cnp.ndarray snake_colors, cnp.ndarray actions,
            cnp.ndarray rewards, int food, int vision,
            int max_snake_length, bint leave_corpse_on_death,
            float reward_food, float reward_corpse,
            float reward_death):
        self.grid = grid
        self.observations = observations
        self.snake = snake
        self.snake_lengths = snake_lengths
        self.snake_ptr = snake_ptr
        self.snake_lifetimes = snake_lifetimes
        self.snake_colors = snake_colors
        self.actions = actions
        self.rewards = rewards
        self.num_snakes = snake.shape[0]
        self.width = grid.shape[1]
        self.height = grid.shape[0]
        self.max_snake_length = max_snake_length
        self.food = food
        self.vision = vision
        self.leave_corpse_on_death = leave_corpse_on_death
        self.reward_food = reward_food
        self.reward_corpse = reward_corpse
        self.reward_death = reward_death

    cdef void compute_observations(self):
        cdef int i, head_ptr, head_r, head_c
        for i in range(self.num_snakes):
            head_ptr = self.snake_ptr[i]
            head_r = self.snake[i, head_ptr, 0]
            head_c = self.snake[i, head_ptr, 1]
            self.observations[i] = self.grid[
                head_r - self.vision:head_r + self.vision + 1,
                head_c - self.vision:head_c + self.vision + 1,
            ]

    cdef void spawn_snake(self, int snake_id):
        cdef int head_ptr, head_r, head_c

        # Delete the snake from the grid
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
        self.snake_lifetimes[snake_id] = 0

    cdef void spawn_food(self):
        cdef int r, c, tile
        while True:
            r = rand() % (self.height - 1)
            c = rand() % (self.width - 1)
            tile = self.grid[r, c]
            if tile == EMPTY or tile == CORPSE:
                self.grid[r, c] = FOOD
                return

    cpdef void reset(self):
        self.grid[:self.vision, :] = WALL
        self.grid[:, :self.vision] = WALL
        self.grid[:, self.width-self.vision-1:] = WALL
        self.grid[self.height-self.vision-1:, :] = WALL

        for i in range(self.num_snakes):
            self.spawn_snake(i)

        for i in range(self.food):
            self.spawn_food()

        self.compute_observations()

    cdef void step(self):
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

            tile = self.grid[next_r, next_c]
            if tile >= WALL:
                self.rewards[i] = self.reward_death
                self.spawn_snake(i)
                continue

            head_ptr += 1
            if head_ptr >= self.max_snake_length:
                head_ptr = 0
            
            self.snake[i, head_ptr, 0] = next_r
            self.snake[i, head_ptr, 1] = next_c
            self.snake_ptr[i] = head_ptr
            self.snake_lifetimes[i] += 1

            if tile == FOOD:
                self.rewards[i] = self.reward_food
                self.spawn_food()
                grow = True
            elif tile == CORPSE:
                self.rewards[i] = self.reward_corpse
                grow = True
            else:
                self.rewards[i] = 0.0
                grow = False

            # If the snake actually reaches max len, you can't move it
            if grow and snake_length < self.max_snake_length - 1:
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
