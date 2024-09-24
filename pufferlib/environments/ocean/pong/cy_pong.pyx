cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "pong.h":
    ctypedef struct CPong:
        float* observations
        unsigned int* actions
        float* rewards
        unsigned char* terminals
        float* paddle_yl_yr
        float* ball_x_y
        float* ball_vx_vy
        unsigned int* score_l_r
        float width
        float height
        float paddle_width
        float paddle_height
        float ball_width
        float ball_height
        float paddle_speed
        float ball_initial_speed_x
        float ball_initial_speed_y
        float ball_max_speed_y
        float ball_speed_y_increment
        unsigned int max_score
        float min_paddle_y
        float max_paddle_y
        float paddle_dir
        unsigned int* misc_logging
        int tick
        int n_bounces
        int win
        int frameskip

    ctypedef struct Client

    void init(CPong* env)
    void reset(CPong* env)
    void step(CPong* env)

    Client* make_client(CPong* env)
    void close_client(Client* client)
    void render(Client* client, CPong* env)

cdef class CyPong:
    cdef:
        CPong env
        Client* client
        float width
        float height
        float paddle_width
        float paddle_height
        float ball_width
        float ball_height

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
            cnp.ndarray rewards, cnp.ndarray terminals, cnp.ndarray paddle_yl_yr,
            cnp.ndarray ball_x_y, cnp.ndarray ball_vx_vy, cnp.ndarray score_l_r,
            float width, float height, float paddle_width, float paddle_height,
            float ball_width, float ball_height, float paddle_speed,
            float ball_initial_speed_x, float ball_initial_speed_y,
            float ball_max_speed_y, float ball_speed_y_increment,
            unsigned int max_score, cnp.ndarray misc_logging):

        self.client = NULL
        self.env = CPong(
            observations = <float*> observations.data,
            actions = <unsigned int*> actions.data,
            rewards = <float*> rewards.data,
            terminals = <unsigned char*> terminals.data,
            paddle_yl_yr = <float*> paddle_yl_yr.data,
            ball_x_y = <float*> ball_x_y.data,
            ball_vx_vy = <float*> ball_vx_vy.data,
            score_l_r = <unsigned int*> score_l_r.data,
            width=width,
            height=height,
            paddle_width=paddle_width,
            paddle_height=paddle_height,
            ball_width=ball_width,
            ball_height=ball_height,
            paddle_speed=paddle_speed,
            ball_initial_speed_x=ball_initial_speed_x,
            ball_initial_speed_y=ball_initial_speed_y,
            ball_max_speed_y=ball_max_speed_y,
            ball_speed_y_increment=ball_speed_y_increment,
            max_score=max_score,
            misc_logging = <unsigned int*> misc_logging.data,
            frameskip=4,
        )
        init(&self.env)

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

        free(&self.env)
