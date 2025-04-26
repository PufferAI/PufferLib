cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

# NOTE: Only methods that are really necessary are c_reset, c_step, and c_render
cdef extern from "boids.h":
    # Put into enum to make cython treat NUM_BOIDS as an Object-like macro.
    # Which is basically a constant done by using "#define".
    enum:
        NUM_BOIDS

    ctypedef struct Velocity:
        float x
        float y

    ctypedef struct Boid:
        float x
        float y
        Velocity velocity
        
    ctypedef struct Boids:
        Boid observations[NUM_BOIDS][NUM_BOIDS]
        Velocity actions[NUM_BOIDS]
        float rewards[NUM_BOIDS]
        unsigned char* terminals
        Boid boids[NUM_BOIDS]
        unsigned int num_boids

    ctypedef struct Client:
        float width
        float height

    void c_init(Boids* env)
    void c_reset(Boids* env)
    void c_step(Boids* env, Velocity* action)

    Client* c_make_client(Boids* env)
    void c_close_client(Client* client)
    void c_render(Client* client, Boids* env)

cdef class CyBoids:
    cdef:
        Boids* envs
        Client* client
        int num_envs
        float width
        float height

    def __init__(self, int num_envs, unsigned int num_boids, float[:, :] observations, Velocity[:, :] actions, float[:] rewards, unsigned char[:] terminals):
        self.num_envs = num_envs
        self.client = NULL
        self.envs = <Boids*> calloc(num_envs, sizeof(Boids))

        cdef int indx
        for indx in range(self.num_envs):
            self.envs[indx] = Boids(num_boids=num_boids)
            c_init(&self.envs[indx])

    def reset(self):
        cdef int indx
        for indx in range(self.num_envs):
            c_reset(&self.envs[indx])

    def step(self, actions):
        cdef int indx
        cdef Velocity action
        for indx in range(self.num_envs):
            action = actions[indx] # Converting each action to a Velocity struct
            c_step(&self.envs[indx], &action)

    def render(self):
        cdef Boids* env = &self.envs[0]
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = c_make_client(env)
            os.chdir(cwd)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            c_close_client(self.client)
            self.client = NULL

        free(self.envs)
