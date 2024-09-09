# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

from cpython.list cimport PyList_GET_ITEM
cimport numpy as cnp


cdef extern from "tactical.h":
    ctypedef struct Tactical:
        int num_agents
        unsigned char* observations
        int* actions
        float* rewards

    Tactical* init_tactical()
    void reset(Tactical* env)
    void step(Tactical* env)


def step_all(list envs):
    cdef:
        int n = len(envs)
        int i
    for i in range(n):
        (<CTactical>PyList_GET_ITEM(envs, i)).step()

cdef class CTactical:
    cdef Tactical* env

    def __init__(self, 
                 cnp.ndarray observations,
                 cnp.ndarray rewards,
                 cnp.ndarray actions,):
        env = init_tactical()
        self.env = env

        env.observations = <unsigned char*> observations.data
        env.actions = <int*> actions.data
        env.rewards = <float*> rewards.data

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)
