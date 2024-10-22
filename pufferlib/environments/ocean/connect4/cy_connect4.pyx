cimport numpy as cnp
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint64_t
import os

cdef extern from "connect4.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct CConnect4:
        float* observations
        int* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer;
        Log log;

        uint64_t player_pieces;
        uint64_t env_pieces;

        int piece_width;
        int piece_height;
        int width;
        int height;

    ctypedef struct Client
    void free_cconnect4(CConnect4* env)
    Client* make_client(float width, float height)
    void close_client(Client* client)
    void render(Client* client, CConnect4* env)
    void reset(CConnect4* env)
    void step(CConnect4* env)

cdef class CyConnect4:
    cdef:
        CConnect4* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int width, int height, int piece_width, int piece_height):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <CConnect4*> calloc(num_envs, sizeof(CConnect4))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = CConnect4(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                dones=&terminals[i],
                log_buffer=self.logs,
                player_pieces=0,
                env_pieces=0,
                piece_width=piece_width,
                piece_height=piece_height,
                width=width,
                height=height,
            )

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef CConnect4* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env.width, env.height)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        # TODO: free
        free_cconnect4(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
