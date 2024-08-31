
cdef extern from "snake.h":
    cdef:
        ctypedef struct CSnake
        CSnake* init_csnake(char* grid, int* snake, char* observations, int* snake_lengths,
            int* snake_ptr, int* snake_lifetimes, int* snake_colors, unsigned int* actions,
            float* rewards, int num_snakes, int width, int height, int max_snake_length,
            int food, int vision, bint leave_corpse_on_death, float reward_food,
            float reward_corpse, float reward_death)
        void step_all(CSnake* env)
        void compute_observations(CSnake* env)
        void spawn_snake(CSnake* env, int snake_id)
        void spawn_food(CSnake* env)
        void reset(CSnake* env)
        void step_snake(CSnake* env, int i)
        void step(CSnake* env)
        ctypedef struct Renderer
        Renderer* init_renderer(int cell_size, int width, int height)
        void render_global(Renderer* renderer, CSnake* env)
        void close_renderer(Renderer* renderer)

cimport numpy as cnp

cdef class Snake:
    cdef:
        CSnake *env
        Renderer *renderer
        char[:, :] grid
        char[:, :, :] observations
        int[:, :, :] snake
        int[:] snake_lengths
        int[:] snake_ptr
        int[:] snake_lifetimes
        int[:] snake_colors
        unsigned int[:] actions
        float[:] rewards

    def __init__(self, cnp.ndarray grid, cnp.ndarray snake,
            cnp.ndarray observations, cnp.ndarray snake_lengths,
            cnp.ndarray snake_ptr, cnp.ndarray snake_lifetimes,
            cnp.ndarray snake_colors, cnp.ndarray actions,
            cnp.ndarray rewards, int food, int vision,
            int max_snake_length, bint leave_corpse_on_death,
            float reward_food, float reward_corpse,
            float reward_death):

        cdef:
            char* grid_view = <char*> grid.data
            int* snake_view = <int*> snake.data
            char* observations_view = <char*> observations.data
            int* snake_lengths_view = <int*> snake_lengths.data
            int* snake_ptr_view = <int*> snake_ptr.data
            int* snake_lifetimes_view = <int*> snake_lifetimes.data
            int* snake_colors_view = <int*> snake_colors.data
            unsigned int* actions_view = <unsigned int*> actions.data
            float* rewards_view = <float*> rewards.data
            int num_snakes = snake.shape[0]
            int width = grid.shape[1]
            int height = grid.shape[0]

        self.env = init_csnake(grid_view, snake_view, observations_view, snake_lengths_view,
            snake_ptr_view, snake_lifetimes_view, snake_colors_view, actions_view,
            rewards_view, num_snakes, width, height, max_snake_length,
            food, vision, leave_corpse_on_death, reward_food,
            reward_corpse, reward_death)
        self.renderer = NULL

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

    def render(self, int cell_size=16, int width=80, int height=45):
        if self.renderer == NULL:
            self.renderer = init_renderer(cell_size, width, height)

        render_global(self.renderer, self.env)

    def close(self):
        if self.renderer != NULL:
            close_renderer(self.renderer)
            self.renderer = NULL
