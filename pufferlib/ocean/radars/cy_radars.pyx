cimport numpy as cnp
from libc.stdlib cimport calloc, free

cdef extern from "radars.h":


    ctypedef struct Target:
        float x
        float x_velocity
        float x_acceleration
        float y
        float y_velocity
        float y_acceleration
        float z
        float z_velocity
        float z_acceleration
        float singer_sigma
        float singer_theta
        float priority

    ctypedef struct Radars:
        short* observations
        int* actions
        float* rewards
        unsigned char* terminals
        int tick
        int s_band_t_until_free
        int x_band_t_until_free
        Target* targets
        int initial_targets

    ctypedef struct Client:
        unsigned int px

    void c_reset(Radars* env)
    void c_step(Radars* env)
    Client* make_client(Radars* env)
    void close_client(Client* client)
    void c_render(Client* client, Radars* env)

    int MAX_TRACKERS

cdef class CyRadars:
    cdef:
        Radars* envs
        Client* client
        int num_envs
        int initial_targets


    def __init__(self, short[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs, int initial_targets):

        self.envs = <Radars*> calloc(num_envs, sizeof(Radars))
        self.num_envs = num_envs
        self.client = NULL

        cdef int i

        for i in range(num_envs):
            self.envs[i].targets = <Target*> calloc(MAX_TRACKERS, sizeof(Target))

            self.envs[i].observations = &observations[i, 0]
            self.envs[i].actions = &actions[i]
            self.envs[i].rewards = &rewards[i]
            self.envs[i].terminals = &terminals[i]
            self.envs[i].tick = 0 
            self.envs[i].s_band_t_until_free = 0
            self.envs[i].x_band_t_until_free = 0
            self.envs[i].initial_targets = initial_targets
        
    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef Radars* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)
