cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "go.h":
    ctypedef struct CGo:
        float* observations
        unsigned char* actions
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
            actions=<unsigned char*> actions.data,
            rewards=<float*> rewards.data,
            dones=<unsigned char*> dones.data,
            width=width,
            height=height,
            grid_size=grid_size,
            board_width=board_width,
            board_height=board_height,
            grid_square_size=grid_square_size,
            moves_made=moves_made,
            komi=komi
        )
        init(&self.env)

    def reset(self):
        reset(&self.env)

    def step(self):
        step(&self.env)

    def render(self):
        render(&self.env)

    def close(self):
        close_client(&self.env)

        free_initialized(&self.env)
