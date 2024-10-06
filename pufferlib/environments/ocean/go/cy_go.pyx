cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "go.h":
    ctypedef struct CGo:
        float* observations
        unsigned short* actions
        float* rewards
        unsigned char* dones
        float score
        float episode_return
        int width
        int height
        int grid_size
        int board_width
        int board_height
        int grid_square_size
        int moves_made
        int client_initialized
        float komi

    void init(CGo* env)
    void free_initialized(CGo* env)

    void init_client(CGo* env)
    void close_client(CGo* env)
    void render(CGo* env)
    void reset(CGo* env)
    void step(CGo* env)

cdef class CyGo:
    cdef:
        CGo env
    def __init__(self, cnp.ndarray actions,
            cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray dones,
            int width, int height, int grid_size, int board_width, int board_height, int grid_square_size, int moves_made, float komi):
        self.env = CGo(
            observations=<float*> observations.data,
            actions=<unsigned short*> actions.data,
            rewards=<float*> rewards.data,
            dones=<unsigned char*> dones.data,
            width=width,
            height=height,
            grid_size=grid_size,
            board_width=board_width,
            board_height=board_height,
            grid_square_size=grid_square_size,
            moves_made=moves_made,
            komi=komi,
            client_initialized=0
        )
        init(&self.env)

    def reset(self):
        reset(&self.env)

    def step(self):
        step(&self.env)

    def render(self):
        if not self.env.client_initialized:
            init_client(&self.env)
            self.env.client_initialized = 1
        render(&self.env)

    def close(self):
        if self.env.client_initialized:
            close_client(&self.env)
            self.env.client_initialized = 0
        free_initialized(&self.env)
