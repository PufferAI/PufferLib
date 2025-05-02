from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t
cdef extern from "raylib.h":
    ctypedef struct Texture2D:
        unsigned int id
        int width
        int height
        int mipmaps
        int format

cdef extern from "cpr.h":

    int LOG_BUFFER_SIZE

    ctypedef struct Log: 
        float perf
        float score
        float episode_return
        float episode_length
        float moves
        float food_nb
        float agents_alive
        float alive_steps
        float n

    ctypedef struct LogBuffer: 
        Log logs 
        int length 
        int idx 

    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Agent: 
        int r 
        int c 
        int id 
        int hp 
        int direction 

    ctypedef struct FoodList:
        int *indexes 
        int size 

    ctypedef struct CCpr:
        int width
        int height
        int num_agents

        int vision
        int vision_window
        int obs_size

        int tick

        float reward_food
        float reward_move

        unsigned char *grid
        unsigned char *observations
        int *actions
        float *rewards
        unsigned char *terminals
        unsigned char *masks
        unsigned char *truncations

        Agent *agents

        LogBuffer *log_buffer
        Log *log

        uint8_t *interactive_food_agent_count
        float interactive_food_reward

        FoodList *foods
        float food_base_spawn_rate

    ctypedef struct Renderer: 
        int cell_size
        int width
        int height
        Texture2D puffer

    void init_ccpr(CCpr *env)
    void c_reset(CCpr *env)
    Renderer *init_renderer(int cell_size, int width, int height)
    void c_step(CCpr *env)
    void c_render(Renderer *renderer, CCpr *env)
    void close_renderer(Renderer *renderer)
    void free_CCpr(CCpr *env)

cdef class CyEnv:
    cdef:
        CCpr *envs
        Renderer *renderer
        LogBuffer *logs 
        int num_envs 

    def __init__(
        self, unsigned char[:,:] observations, int[:] actions, float[:] rewards,
        unsigned char[:] terminals, unsigned char[:] truncations, unsigned char[:] masks,
        list widths, list heights, list num_agents,int vision, 
        float reward_food,float interactive_food_reward,float reward_move, float food_base_spawn_rate
    ) -> None:
        self.num_envs = len(num_agents)
        self.envs = <CCpr*>calloc(self.num_envs, sizeof(CCpr))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i 
        cdef int n = 0 
        for i in range(self.num_envs):
            self.envs[i] = CCpr(
                observations = &observations[n,0],
                actions=&actions[n],
                rewards=&rewards[n],
                terminals=&terminals[n],
                truncations=&truncations[n],
                masks=&masks[n],
                log_buffer=self.logs,
                width=widths[i],
                height=heights[i],
                num_agents = num_agents[i],
                vision=vision,
                reward_food=reward_food,
                interactive_food_reward=interactive_food_reward,
                reward_move=reward_move,
                food_base_spawn_rate=food_base_spawn_rate,
            )
            init_ccpr(&self.envs[i])
            n += num_agents[i]

    def step(self): 
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])
        
    def render(self, cell_size=32): 
        cdef CCpr* env = &self.envs[0]
        if self.renderer == NULL:
            self.renderer = init_renderer(cell_size, env.width, env.height)
        c_render(self.renderer, env)

    def close(self):
        if self.renderer != NULL:
            close_renderer(self.renderer) 
            self.renderer = NULL

        cdef int i 
        for i in range(self.num_envs):
            free_CCpr(&self.envs[i])
        free(self.envs)

    def reset(self): 
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
