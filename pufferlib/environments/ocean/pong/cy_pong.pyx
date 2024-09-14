cimport numpy as cnp
from libc.stdlib cimport free

cdef extern from "pong.h":
    ctypedef struct CMyPong:
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

    CMyPong* init_cmy_pong(float* observations, unsigned int* actions,
            float* rewards, unsigned char* terminals, float* paddle_yl_yr, float* ball_x_y,
            float* ball_vx_vy, unsigned int* score_l_r, float width, float height,
            float paddle_width, float paddle_height, float ball_width, float ball_height,
            float paddle_speed, float ball_initial_speed_x, float ball_initial_speed_y,
            float ball_max_speed_y, float ball_speed_y_increment, unsigned int max_score,
            unsigned int* misc_logging)
 
    Client* make_client(float width, float height, float paddle_width,
        float paddle_height, float ball_width, float ball_height)
    void close_client(Client* client)
    void render(Client* client, CMyPong* env)
    void reset(CMyPong* env)
    void step(CMyPong* env)

cdef class CPong:
    cdef:
        CMyPong* env
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

        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height
        self.ball_width = ball_width
        self.ball_height = ball_height

        self.client = NULL

        self.env = init_cmy_pong(<float*> observations.data, <unsigned int*> actions.data,
            <float*> rewards.data, <unsigned char*> terminals.data,
            <float*> paddle_yl_yr.data, <float*> ball_x_y.data, <float*> ball_vx_vy.data,
            <unsigned int*> score_l_r.data, width, height,
            paddle_width, paddle_height, ball_width, ball_height, paddle_speed,
            ball_initial_speed_x, ball_initial_speed_y, ball_max_speed_y,
            ball_speed_y_increment, max_score, <unsigned int*> misc_logging.data)

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

    def render(self):
        if self.client == NULL:
            self.client = make_client(self.width, self.height, self.paddle_width,
                self.paddle_height, self.ball_width, self.ball_height)

        render(self.client, self.env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.env)
