from libc.stdlib cimport calloc, free, rand, srand
import numpy as np
cimport numpy as cnp

cdef extern from "stdbool.h":
    ctypedef bint bool

include "codeball.pxd"
include "renderer.pxd"


cdef extern from "codeball.h":
    cdef double ROBOT_MAX_JUMP_SPEED
    cdef double ROBOT_MAX_GROUND_SPEED

cdef class CyCodeBall:
    cdef CodeBall* envs
    cdef int num_envs
    cdef int n_robots
    cdef int max_steps
    cdef double reward_mul
    cdef float[:, :] action_buffer
    cdef float[:] reward_buffer
    cdef float[:, :, :] observation_buffer
    cdef bool[:] terminal_buffer
    cdef bool[:] truncate_buffer
    cdef LogBuffer* log_aggregator
    cdef Client* client
    cdef bool is_single
    cdef int n_agents

    def __init__(self,
        int num_envs, int n_robots, int n_nitros, int frame_skip, double reward_mul, int max_steps,
        bool is_single,
        float goal_scored_reward, float loiter_penalty, float ball_reward,
        int baseline,
        float [:, :, :] observations,
        float [:, :] actions, float [:] rewards, bool [:] terminals, bool [:] truncations
    ):
        self.is_single = is_single
        self.num_envs = num_envs
        self.n_robots = n_robots
        self.max_steps = max_steps
        self.reward_mul = reward_mul
        self.envs = <CodeBall*>calloc(num_envs, sizeof(CodeBall))
        self.observation_buffer = observations
        self.action_buffer = actions
        self.reward_buffer = rewards
        self.terminal_buffer = terminals
        self.truncate_buffer = truncations

        if is_single:
            self.n_agents = self.n_robots // 2
        else:
            self.n_agents = self.n_robots
        cdef int i
        for i in range(num_envs):
            self.envs[i].n_robots = n_robots
            self.envs[i].n_nitros = n_nitros
            self.envs[i].frame_skip = frame_skip
            self.envs[i].is_single = is_single
            self.envs[i].goal_scored_reward = goal_scored_reward
            self.envs[i].loiter_penalty = loiter_penalty
            self.envs[i].ball_reward = ball_reward
            self.envs[i].actions = &self.action_buffer[i * self.n_agents, 0]
            if baseline - 1 == 0:
                self.envs[i].baseline = DO_NOTHING
            elif baseline - 1 == 1:
                self.envs[i].baseline = RANDOM_ACTIONS
            else:
                self.envs[i].baseline = RUN_TO_BALL
            allocate(&self.envs[i])
        
        self.log_aggregator = allocate_logbuffer(self.num_envs)

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])
        self._observe()

    def _observe(self):
        cdef int i, j, source_idx
        cdef float rew
        for i in range(self.num_envs):
            make_observation(&self.envs[0], &self.observation_buffer[i * self.n_agents, 0, 0])
        for i in range(self.num_envs):
            for j in range(self.n_agents):
                self.terminal_buffer[i * self.n_agents + j] = self.envs[i].terminal
                source_idx = j
                if self.is_single:
                    source_idx = j * 2
                self.reward_buffer[i * self.n_agents + j] = self.envs[i].rewards[source_idx] * self.reward_mul

    def log_nth(self, int i):
        cdef Log log
        log = aggregate(self.envs[i].log_buffer)
        return log

    def log(self):
        cdef int i
        cdef Log log
        self.log_aggregator.idx = 0
        self.log_aggregator.length = 0
        for i in range(self.num_envs):
            log = aggregate(self.envs[i].log_buffer)
            add_log(self.log_aggregator, &log)
        log = aggregate(self.log_aggregator)
        return log

    def render(self):
        if self.client == NULL:
            self.client = make_client()
        render(self.client, &self.envs[0])

    def step(self,):
        cdef int i, j, target_idx
        cdef float vel_x, vel_z
        for i in range(self.num_envs):
            if self.envs[i].tick >= self.max_steps:
                self.truncate_buffer[i] = True
                reset(&self.envs[i])
            else:
                self.truncate_buffer[i] = False

        for i in range(self.num_envs):
            step(&self.envs[i])
        
        self._observe()

    def close(self):
        cdef int i
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL
        for i in range(self.num_envs):
            free_allocated(&self.envs[i])
        free(self.envs)
        free_logbuffer(self.log_aggregator)
