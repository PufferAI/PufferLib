cimport numpy as cnp
from libc.stdlib cimport calloc, free  # Use calloc for zero-initialized allocation
from libc.stdint cimport uint64_t

cdef extern from "trash_pickup.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float trash_collected;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct CTrashPickupEnv:
        char* observations
        int* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer

        int grid_size
        int num_agents
        int num_trash
        int num_bins
        int max_steps
        int agent_sight_range


    ctypedef struct Client

    void initialize_env(CTrashPickupEnv* env)
    void free_allocated(CTrashPickupEnv* env)

    Client* make_client(CTrashPickupEnv* env)
    void close_client(Client* client)
    void c_render(Client* client, CTrashPickupEnv* env) 
    void c_reset(CTrashPickupEnv* env)
    void c_step(CTrashPickupEnv* env)

cdef class CyTrashPickup:
    cdef:
        CTrashPickupEnv* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, char[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs, 
            int num_agents=3, int grid_size=10, int num_trash=15, 
            int num_bins=2, int max_steps=300, int agent_sight_range=5):
        self.num_envs = num_envs
        self.envs = <CTrashPickupEnv*>calloc(num_envs, sizeof(CTrashPickupEnv))
        if self.envs == NULL:
            raise MemoryError("Failed to allocate memory for CTrashPickupEnv")
        self.client = NULL

        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int inc = num_agents
        
        cdef int i
        for i in range(num_envs):
            self.envs[i] = CTrashPickupEnv(
                observations=&observations[inc*i, 0],
                actions=&actions[inc*i],
                rewards=&rewards[inc*i],
                dones=&terminals[inc*i],
                log_buffer=self.logs, 
                grid_size=grid_size, 
                num_agents=num_agents,
                num_trash=num_trash, 
                num_bins=num_bins, 
                max_steps=max_steps,
                agent_sight_range=agent_sight_range
            )
            initialize_env(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef CTrashPickupEnv* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
