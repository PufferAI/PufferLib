cimport numpy as cnp
from libc.stdlib cimport calloc, free

cdef extern from "squared.h":
    ctypedef struct Client:
        pass

    ctypedef struct Squared:
        Client* client
        unsigned char* observations
        int* actions
        float* rewards
        unsigned char* terminals
        int size
        int tick
        int r
        int c

    void c_reset(Squared* env)
    void c_step(Squared* env)
    void c_render(Squared* env)


cdef class CySquared:
    cdef:
        Squared* envs
        int num_envs
        int size

    def __init__(self, unsigned char[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs, int size):

        self.envs = <Squared*> calloc(num_envs, sizeof(Squared))
        self.num_envs = num_envs

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Squared(
                observations = &observations[i, 0],
                actions = &actions[i],
                rewards = &rewards[i],
                terminals = &terminals[i],
                size=size,
            )

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef Squared* env = &self.envs[0]
        c_render(env)

    def close(self):
        free(self.envs)
