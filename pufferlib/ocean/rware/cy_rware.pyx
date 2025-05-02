from libc.stdlib cimport calloc, free

cdef extern from "rware.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float perf;
        float score;
        float episode_return;
        float episode_length;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct MovementGraph:
        int* target_positions;
        int* cycle_ids;
        int* weights;
        int num_cycles;

    ctypedef struct CRware:
        float* observations;
        int* actions;
        float* rewards;
        unsigned char* dones;
        LogBuffer* log_buffer;
        Log* logs;
        float* scores;
        int width;
        int height;
        int map_choice;
        int* warehouse_states;
        int num_agents;
        int num_requested_shelves;
        int* agent_locations;
        int* agent_directions;
        int* agent_states;
        int shelves_delivered;
        int human_agent_idx;
        int grid_square_size;
        int* original_shelve_locations;
        MovementGraph* movement_graph;

    ctypedef struct Client

    void init(CRware* env)
    void free_allocated(CRware* env)


    Client* make_client(CRware* env)
    void close_client(Client* client)
    void c_render(Client* client, CRware* env)
    void c_reset(CRware* env)
    void c_step(CRware* env)

cdef class CyRware:
    cdef:
        CRware* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int width, int height, int map_choice, int num_agents, int num_requested_shelves, 
            int grid_square_size, int human_agent_idx):

        self.client = NULL
        self.num_envs = num_envs
        self.envs = <CRware*> calloc(num_envs, sizeof(CRware))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int inc = num_agents

        cdef int i
        for i in range(num_envs):
            self.envs[i] = CRware(
                observations=&observations[inc*i, 0],
                actions=&actions[inc*i],
                rewards=&rewards[inc*i],
                dones=&terminals[inc*i],
                log_buffer=self.logs,
                width=width,
                height=height,
                map_choice=map_choice,
                num_agents=num_agents,
                num_requested_shelves=num_requested_shelves,
                grid_square_size=grid_square_size,
                human_agent_idx=human_agent_idx,
            )
            init(&self.envs[i])
            self.client = NULL

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef CRware* env = &self.envs[0]
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(env)
            os.chdir(cwd)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
