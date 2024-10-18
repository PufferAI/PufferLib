cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

cdef extern from "rocket_lander.h":
    int LOG_BUFFER_SIZE

    ctypedef struct b2WorldId:
        unsigned short index1
        unsigned short revision
    ctypedef struct b2BodyId:
        int index1
        unsigned short revision
        unsigned char world0
    ctypedef struct b2Vec2:
        float x
        float y

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Entity:
        b2BodyId bodyId;
        b2Vec2 extent;

    ctypedef struct Lander:
        float* observations;
        float* actions;
        float* reward;
        unsigned char* terminal;
        unsigned char* truncation;
        LogBuffer* log_buffer;
        Log log;
        int tick;
        b2WorldId world_id;
        b2BodyId barge_id;
        b2BodyId lander_id;
        Entity barge;
        Entity lander;

    ctypedef struct Client

    void init_lander(Lander* env)
    void reset(Lander* env)
    void step(Lander* env)
    void free_lander(Lander* env)

    Client* make_client(Lander* env)
    void render(Client* client, Lander* env)
    void close_client(Client* client)

cdef class CyRocketLander:
    cdef:
        Lander* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
            cnp.ndarray rewards, cnp.ndarray terminals,
            cnp.ndarray truncations, int num_envs):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <Lander*> calloc(num_envs, sizeof(Lander))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef:
            cnp.ndarray observations_i
            cnp.ndarray actions_i
            cnp.ndarray rewards_i
            cnp.ndarray terminals_i
            cnp.ndarray truncations_i

        cdef int i
        for i in range(num_envs):
            observations_i = observations[i:i+1]
            actions_i = actions[i:i+1]
            rewards_i = rewards[i:i+1]
            terminals_i = terminals[i:i+1]
            truncations_i = truncations[i:i+1]
            self.envs[i] = Lander(
                observations = <float*> observations_i.data,
                actions = <float*> actions_i.data,
                reward = <float*> rewards_i.data,
                terminal = <unsigned char*> terminals_i.data,
                truncation = <unsigned char*> truncations_i.data,
                log_buffer=self.logs,
            )
            init_lander(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef Lander* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        cdef int i
        for i in range(self.num_envs):
            free_lander(&self.envs[i])
        free(self.envs)
        free(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
