# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=True

from libc.stdlib cimport calloc, free

cdef extern from "grid.h":
    ctypedef struct Agent:
        float y;
        float x;
        float spawn_y;
        float spawn_x;
        int color;
        int direction;
        int held_object;
        int keys[6];

    ctypedef struct Env:
        int width;
        int height;
        int num_agents;
        int horizon;
        int vision;
        float speed;
        bint discretize;
        int obs_size;
        int tick;
        float episode_return;

        unsigned char* grid;
        Agent* agents;
        unsigned char* observations;
        unsigned int* actions;
        float* rewards;
        float* dones;

    cdef:
        #Env* init_grid(
        #        unsigned char* observations, unsigned int* actions, float* rewards, float* dones,
        #        int width, int height, int num_agents, int horizon,
        #        int vision, float speed, bint discretize)
        Env** make_locked_room_env(unsigned char* observations,
            unsigned int* actions, float* rewards, float* dones)
        void reset_locked_room(Env* env)
        bint step(Env* env)
        void free_envs(Env** env, int num_envs)
        Env** make_locked_rooms(unsigned char* observations,
            unsigned int* actions, float* rewards, float* dones, int num_envs)
        ctypedef struct Renderer
        Renderer* init_renderer(int cell_size, int width, int height)
        void render_global(Renderer*erenderer, Env* env)
        void close_renderer(Renderer* renderer)

cimport numpy as cnp

cdef class CGrid:
    cdef:
        Env **envs
        Renderer *renderer
        int num_envs
        int num_finished
        float sum_returns

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
                 cnp.ndarray rewards, cnp.ndarray dones, int num_envs):

        #self.env = init_grid(<unsigned char*> observations.data, <unsigned int*> actions.data,
        #    <float*> rewards.data, <float*> dones.data, width, height, num_agents, horizon,
        #    vision, speed, discretize)
        self.envs = make_locked_rooms(
            <unsigned char*> observations.data, <unsigned int*> actions.data,
            <float*> rewards.data, <float*> dones.data, num_envs)

        self.renderer = NULL
        self.num_envs = num_envs

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset_locked_room(self.envs[i])

    def step(self):
        cdef:
            int i
            bint done
        
        for i in range(self.num_envs):
            done = step(self.envs[i])
            if done:
                self.num_finished += 1
                self.sum_returns += self.envs[i].episode_return
                reset_locked_room(self.envs[i])

    def get_returns(self):
        cdef float returns = self.sum_returns / self.num_finished
        self.sum_returns = 0
        self.num_finished = 0
        return returns

    def has_key(self):
        cdef int num_keys = 0
        cdef int i
        for i in range(self.num_envs):
            if self.envs[i].agents[0].keys[5] == 1:
                num_keys += 1

        return num_keys

    def render(self, int cell_size=16, int width=80, int height=45):
        if self.renderer == NULL:
            import os
            path = os.path.abspath(os.getcwd())
            print(path)
            c_path = os.path.join(os.sep, *__file__.split('/')[:-1])
            print(c_path)
            os.chdir(c_path)
            self.renderer = init_renderer(cell_size, width, height)
            os.chdir(path)

        render_global(self.renderer, self.envs[0])

    def close(self):
        if self.renderer != NULL:
            close_renderer(self.renderer)
            self.renderer = NULL

        free_envs(self.envs, self.num_envs)
