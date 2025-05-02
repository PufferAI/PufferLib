cimport numpy as cnp
from libc.stdlib cimport calloc, free

cdef extern from "snake.h":
    cdef:
        int LOG_BUFFER_SIZE

        ctypedef struct Log:
            float perf
            float score
            float episode_return
            float episode_length

        ctypedef struct LogBuffer
        LogBuffer* allocate_logbuffer(int)
        void free_logbuffer(LogBuffer*)
        Log aggregate_and_clear(LogBuffer*)

        ctypedef struct CSnake:
            char* observations
            int* actions
            float* rewards
            unsigned char* terminals
            LogBuffer* log_buffer
            Log* logs
            char* grid
            int* snake
            int* snake_lengths
            int* snake_ptr
            int* snake_lifetimes
            int* snake_colors
            int num_snakes
            int width
            int height
            int max_snake_length
            int food
            int vision
            int window
            int obs_size
            unsigned char leave_corpse_on_death
            float reward_food
            float reward_corpse
            float reward_death

        ctypedef struct CSnake
        void init_csnake(CSnake* env)
        void step_all(CSnake* env)
        void compute_observations(CSnake* env)
        void spawn_snake(CSnake* env, int snake_id)
        void spawn_food(CSnake* env)
        void c_reset(CSnake* env)
        void step_snake(CSnake* env, int i)
        void c_step(CSnake* env)
        ctypedef struct Client
        Client* make_client(int cell_size, int width, int height)
        void c_render(Client* client, CSnake* env)
        void close_client(Client* client)

cdef class CySnake:
    cdef:
        CSnake *envs
        Client* client
        LogBuffer* logs
        int num_envs
        
    def __init__(self, char[:, :, :] observations, int[:] actions,
             float[:] rewards, unsigned char[:] terminals,
             list widths, list heights, list num_snakes,
             list num_food, int vision, int max_snake_length,
             bint leave_corpse_on_death, float reward_food,
             float reward_corpse, float reward_death):

        self.num_envs = len(num_snakes)
        self.envs = <CSnake*>calloc(self.num_envs, sizeof(CSnake))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        self.client = NULL

        cdef int i
        cdef int n = 0
        for i in range(self.num_envs):
            self.envs[i] = CSnake(
                observations=&observations[n, 0, 0],
                actions=&actions[n],
                rewards=&rewards[n],
                terminals=&terminals[n],
                log_buffer=self.logs,
                width=widths[i],
                height=heights[i],
                num_snakes=num_snakes[i],
                food=num_food[i],
                vision=vision,
                max_snake_length=max_snake_length,
                leave_corpse_on_death=leave_corpse_on_death,
                reward_food=reward_food,
                reward_corpse=reward_corpse,
                reward_death=reward_death,
            )
            init_csnake(&self.envs[i])
            n += num_snakes[i]

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self, cell_size=8):
        cdef CSnake* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(cell_size, env.width, env.height)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        # TODO: free
        #free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
