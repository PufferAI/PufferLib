cimport numpy as cnp
from libc.stdlib cimport calloc, free

cdef extern from "snake.h":
    cdef:
        int LOG_BUFFER_SIZE

        ctypedef struct Log:
            float episode_return;
            float episode_length;
            float score;

        ctypedef struct LogBuffer
        LogBuffer* allocate_logbuffer(int)
        void free_logbuffer(LogBuffer*)
        Log aggregate_and_clear(LogBuffer*)

        ctypedef struct CSnake:
            char* observations
            unsigned int* actions
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
        void reset(CSnake* env)
        void step_snake(CSnake* env, int i)
        void step(CSnake* env)
        ctypedef struct Client
        Client* make_client(int cell_size, int width, int height)
        void render(Client* client, CSnake* env)
        void close_client(Client* client)

cdef class Snake:
    cdef:
        CSnake *envs
        Client* client
        LogBuffer* logs
        int num_envs
        
    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
             cnp.ndarray rewards, cnp.ndarray terminals,
             list widths, list heights, list num_snakes,
             list num_food, int vision, int max_snake_length,
             bint leave_corpse_on_death, float reward_food,
             float reward_corpse, float reward_death):

        self.num_envs = len(num_snakes)
        self.envs = <CSnake*>calloc(self.num_envs, sizeof(CSnake*))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        self.client = NULL

        cdef:
            cnp.ndarray observations_i
            cnp.ndarray actions_i
            cnp.ndarray rewards_i
            cnp.ndarray terminals_i

        cdef int i
        cdef int n = 0
        for i in range(self.num_envs):
            observations_i = observations[n:n+num_snakes[i]]
            actions_i = actions[n:n+num_snakes[i]]
            rewards_i = rewards[n:n+num_snakes[i]]
            terminals_i = terminals[n:n+num_snakes[i]]
            n += num_snakes[i]

            self.envs[i] = CSnake(
                observations = <char*> observations_i.data,
                actions = <unsigned int*> actions_i.data,
                rewards = <float*> rewards_i.data,
                terminals = <unsigned char*> terminals_i.data,
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

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef CSnake* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(8, env.width, env.height)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        # TODO: free
        #free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
