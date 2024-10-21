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

    void free_tactical(Tactical* env)

    ctypedef struct GameRenderer

    GameRenderer* init_game_renderer(Tactical* env)
    int render_game(GameRenderer* renderer, Tactical* env)
    void close_game_renderer(GameRenderer* renderer)


def step_all(list envs):
    cdef:
        int n = len(envs)
        int i
    for i in range(n):
        (<CTactical>PyList_GET_ITEM(envs, i)).step()

cdef class CTactical:
    cdef Tactical* env
    cdef GameRenderer* renderer

    def __init__(self, 
                 cnp.ndarray observations,
                 cnp.ndarray rewards,
                 cnp.ndarray actions,):
        env = init_tactical()
        self.env = env

        env.observations = <unsigned char*> observations.data
        env.actions = <int*> actions.data
        env.rewards = <float*> rewards.data

        self.renderer = NULL

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

    def render(self):
        if self.renderer == NULL:
            import os
            path = os.path.abspath(os.getcwd())
            print(path)
            c_path = os.path.join(os.sep, *__file__.split('/')[:-1])
            print(c_path)
            os.chdir(c_path)
            self.renderer = init_game_renderer(self.env)
            os.chdir(path)

        return render_game(self.renderer, self.env)

    def close(self):
        if self.renderer != NULL:
            close_game_renderer(self.renderer)
            self.renderer = NULL

        free_tactical(self.env)
