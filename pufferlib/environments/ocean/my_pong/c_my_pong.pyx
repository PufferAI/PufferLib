# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: profile=False

from cpython.list cimport PyList_GET_ITEM
cimport numpy as cnp
from libc.stdlib cimport rand
from libc.math cimport pi, fmin, fmax, abs

def step_all(list envs):
    cdef int n = len(envs)
    for i in range(n):
        (<CMyPong>PyList_GET_ITEM(envs, i)).step()

cdef class CMyPong:
    cdef:
        float[:] observations
        unsigned int[:] actions
        float[:] rewards
        unsigned char[:] terminals  # TODO uint8, because bool doesn't get converted to bint by default
        float[:] paddle_yl_yr
        float[:] ball_x_y
        float[:] ball_vx_vy
        unsigned int[:] score_l_r
        float width, height
        float paddle_width, paddle_height
        float ball_width, ball_height
        float paddle_speed, ball_initial_speed_x, ball_initial_speed_y
        float ball_max_speed_y, ball_speed_y_increment
        unsigned int max_score
        float min_paddle_y, max_paddle_y
        float paddle_dir

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
            cnp.ndarray rewards, cnp.ndarray terminals,
            cnp.ndarray paddle_yl_yr,
            cnp.ndarray ball_x_y, cnp.ndarray ball_vx_vy,
            cnp.ndarray score_l_r, float width, float height,
            float paddle_width, float paddle_height, float ball_width, float ball_height,
            float paddle_speed, float ball_initial_speed_x, float ball_initial_speed_y,
            float ball_max_speed_y, float ball_speed_y_increment, unsigned int max_score
            ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.paddle_yl_yr = paddle_yl_yr
        self.ball_x_y = ball_x_y
        self.ball_vx_vy = ball_vx_vy
        self.score_l_r = score_l_r
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.paddle_height = paddle_height        
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.paddle_speed = paddle_speed
        self.ball_initial_speed_x = ball_initial_speed_x
        self.ball_initial_speed_y = ball_initial_speed_y
        self.ball_max_speed_y = ball_max_speed_y
        self.ball_speed_y_increment = ball_speed_y_increment
        self.max_score = max_score

        # precompute
        self.min_paddle_y = - self.paddle_height / 2
        self.max_paddle_y = self.height - self.paddle_height / 2
        
        self.paddle_dir = 0

    cdef void compute_observations(self):
        # normalize as floats between 0 and 1
        # TODO: check how much faster it is to pass them as unsigned ints (as in moba)
        self.observations[0] = (self.paddle_yl_yr[0] - self.min_paddle_y) / (self.max_paddle_y - self.min_paddle_y)
        self.observations[1] = (self.paddle_yl_yr[1] - self.min_paddle_y) / (self.max_paddle_y - self.min_paddle_y)
        self.observations[2] = self.ball_x_y[0] / self.width
        self.observations[3] = self.ball_x_y[1] / self.height
        self.observations[4] = (self.ball_vx_vy[0] + self.ball_initial_speed_x) / (2 * self.ball_initial_speed_x)
        self.observations[5] = (self.ball_vx_vy[1] + self.ball_max_speed_y) / (2 * self.ball_max_speed_y)
        self.observations[6] = self.score_l_r[0] / self.max_score
        self.observations[7] = self.score_l_r[1] / self.max_score
        
    cpdef void reset_round(self):
        self.paddle_yl_yr[0] = self.height / 2 - self.paddle_height / 2
        self.paddle_yl_yr[1] = self.height / 2 - self.paddle_height / 2
        self.ball_x_y[0] = self.width / 5
        self.ball_x_y[1] = self.height / 2 - self.ball_height / 2
        self.ball_vx_vy[0] = self.ball_initial_speed_x
        self.ball_vx_vy[1] = (rand() % 2 - 1) * self.ball_initial_speed_y

    cpdef void reset(self):
        self.reset_round()
        self.score_l_r[0] = 0
        self.score_l_r[1] = 0
        self.compute_observations()

    cdef void step(self):
        cdef:
            int i
            unsigned int act
            float opp_paddle_delta

        # TODO i'm not setting any dones here, so it will bootstrap into the next round but it shouldn't
        self.rewards[0] = 0
        self.terminals[0] = 0

        # move ego paddle
        act = self.actions[0]
        self.paddle_dir = 0
        if act == 0:  # still
            self.paddle_dir = 0
        elif act == 1:  # up
            self.paddle_dir = 1
        elif act == 2:  # down
            self.paddle_dir = -1
        self.paddle_yl_yr[1] += self.paddle_speed * self.paddle_dir
        
        # move opponent paddle
        opp_paddle_delta = self.ball_x_y[1] - (self.paddle_yl_yr[0] + self.paddle_height / 2)
        opp_paddle_delta = fmin(fmax(opp_paddle_delta, -self.paddle_speed), self.paddle_speed)
        self.paddle_yl_yr[0] += opp_paddle_delta

        # clip paddles
        self.paddle_yl_yr[1] = fmin(fmax(
            self.paddle_yl_yr[1], self.min_paddle_y), self.max_paddle_y)
        self.paddle_yl_yr[0] = fmin(fmax(
            self.paddle_yl_yr[0], self.min_paddle_y), self.max_paddle_y)

        # move ball
        self.ball_x_y[0] += self.ball_vx_vy[0]
        self.ball_x_y[1] += self.ball_vx_vy[1]

        # handle collision with top & bottom walls
        if self.ball_x_y[1] < 0 or self.ball_x_y[1] + self.ball_height > self.height:
            self.ball_vx_vy[1] = -self.ball_vx_vy[1]

        # handle collision on left
        if self.ball_x_y[0] < 0:
            if self.ball_x_y[1] + self.ball_height > self.paddle_yl_yr[0] and \
                self.ball_x_y[1] < self.paddle_yl_yr[0] + self.paddle_height:
                # collision with paddle
                self.ball_vx_vy[0] = -self.ball_vx_vy[0]
            else:
                # collision with wall
                self.score_l_r[1] += 1
                self.rewards[0] = 50.0
                self.terminals[0] = 1
                self.reset_round()
        
        # handle collision on right (TODO duplicated code)
        if self.ball_x_y[0] + self.ball_width > self.width:
            if self.ball_x_y[1] + self.ball_height > self.paddle_yl_yr[1] and \
                self.ball_x_y[1] < self.paddle_yl_yr[1] + self.paddle_height:
                # collision with paddle
                self.ball_vx_vy[0] = -self.ball_vx_vy[0]
                self.rewards[0] = 10.0
                # ball speed change
                self.ball_vx_vy[1] += self.ball_speed_y_increment * self.paddle_dir
                self.ball_vx_vy[1] = fmin(fmax(self.ball_vx_vy[1], -self.ball_max_speed_y), self.ball_max_speed_y)
                if abs(self.ball_vx_vy[1]) < 0.01:  # we dont want a horizontal ball
                    self.ball_vx_vy[1] = self.ball_speed_y_increment
            else:
                # collision with wall
                self.score_l_r[0] += 1
                self.rewards[0] = -50.0
                if self.score_l_r[0] == self.max_score:
                    self.terminals[0] = 1
                    # it seems the training code doesn't reset automatically on done -> reset here
                    # TODO could it be bootstrapping when it shouldn't be?
                    self.reset()
                else:
                    self.reset_round()

        # clip ball
        self.ball_x_y[0] = fmin(fmax(self.ball_x_y[0], 0), self.width - self.ball_width)
        self.ball_x_y[1] = fmin(fmax(self.ball_x_y[1], 0), self.height - self.ball_height)

        self.compute_observations()


