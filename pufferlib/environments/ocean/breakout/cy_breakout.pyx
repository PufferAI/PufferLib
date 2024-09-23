cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "breakout.h":
    ctypedef struct CBreakout:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* dones
        int score
        float episode_return
        float paddle_x
        float paddle_y
        float ball_x
        float ball_y
        float ball_vx
        float ball_vy
        float* brick_x
        float* brick_y
        float* brick_states
        int balls_fired
        float paddle_width
        float paddle_height
        float ball_speed
        int hits
        int width
        int height
        int num_bricks
        int brick_rows
        int brick_cols
        int ball_width
        int ball_height
        int brick_width
        int brick_height
        int num_balls
        int frameskip

    ctypedef struct Client

    void init(CBreakout* env)
    void free_initialized(CBreakout* env)

    Client* make_client(CBreakout* env)
    void close_client(Client* client)
    void render(Client* client, CBreakout* env)
    void reset(CBreakout* env)
    void step(CBreakout* env)

cdef class CyBreakout:
    cdef:
        CBreakout env
        Client* client

    def __init__(self, int frameskip, cnp.ndarray actions,
            cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray dones,
            int width, int height, float paddle_width, float paddle_height,
            int ball_width, int ball_height, int brick_width, int brick_height,
            int brick_rows, int brick_cols):
        self.env = CBreakout(
            observations=<float*> observations.data,
            actions=<unsigned char*> actions.data,
            rewards=<float*> rewards.data,
            dones=<unsigned char*> dones.data,
            width=width,
            height=height,
            paddle_width=paddle_width,
            paddle_height=paddle_height,
            ball_width=ball_width,
            ball_height=ball_height,
            brick_width=brick_width,
            brick_height=brick_height,
            brick_rows=brick_rows,
            brick_cols=brick_cols,
            frameskip=frameskip,
        )
        init(&self.env)
        self.client = NULL

    def reset(self):
        reset(&self.env)

    def step(self):
        step(&self.env)

    def render(self):
        if self.client == NULL:
            self.client = make_client(&self.env)

        render(self.client, &self.env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free_initialized(&self.env)
