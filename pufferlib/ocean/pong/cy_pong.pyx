cimport numpy as cnp
from libc.stdlib cimport calloc, free
import os

cdef extern from "pong.h":
    ctypedef char* LOG_KEYS[];
    ctypedef struct Client:
        pass

    ctypedef struct Pong:
        Client* client;
        float log[4];
        float* observations;
        float* actions;
        float* rewards;
        unsigned char* terminals;
        float paddle_yl;
        float paddle_yr;
        float ball_x;
        float ball_y;
        float ball_vx;
        float ball_vy;
        unsigned int score_l;
        unsigned int score_r;
        float width;
        float height;
        float paddle_width;
        float paddle_height;
        float ball_width;
        float ball_height;
        float paddle_speed;
        float ball_initial_speed_x;
        float ball_initial_speed_y;
        float ball_max_speed_y;
        float ball_speed_y_increment;
        unsigned int max_score;
        float min_paddle_y;
        float max_paddle_y;
        float paddle_dir;
        int tick;
        int n_bounces;
        int win;
        int frameskip;
        int continuous;

    void init(Pong* env)
    void c_reset(Pong* env)
    void c_step(Pong* env)

    void c_render(Pong* env)

cdef class CyPong:
    cdef:
        Pong* envs
        int num_envs
        float width
        float height
        float paddle_width
        float paddle_height
        float ball_width
        float ball_height

    def __init__(self, float[:, :] observations, float[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            float width, float height, float paddle_width, float paddle_height,
            float ball_width, float ball_height, float paddle_speed,
            float ball_initial_speed_x, float ball_initial_speed_y,
            float ball_max_speed_y, float ball_speed_y_increment,
            unsigned int max_score, int frameskip, int continuous):

        self.num_envs = num_envs
        self.envs = <Pong*> calloc(num_envs, sizeof(Pong))

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Pong(
                observations = &observations[i, 0],
                actions = &actions[i],
                rewards = &rewards[i],
                terminals = &terminals[i],
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
                frameskip=frameskip,
                continuous=continuous,
            )
            init(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
    
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef Pong* env = &self.envs[0]
        c_render(env)

    def close(self):
        free(self.envs)
