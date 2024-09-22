import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
import random

cnp.import_array()

cdef struct CStochasticEnv:
    float p
    int horizon
    int tick
    int count
    int action

cdef CStochasticEnv* init_stochastic_env(float p, int horizon):
    cdef CStochasticEnv* env = <CStochasticEnv*> malloc(sizeof(CStochasticEnv))
    env.p = p
    env.horizon = horizon
    env.tick = 0
    env.count = 0
    env.action = 0
    return env

cdef void reset(CStochasticEnv* env):
    env.tick = 0
    env.count = 0
    env.action = 0

cdef tuple step(CStochasticEnv* env, int action):
    assert 0 <= action <= 1
    env.tick += 1
    if action == 0:
        env.count += 1
    env.action = action

    terminal = env.tick == env.horizon
    atn0_frac = env.count / env.tick
    proximity_to_p = 1 - (env.p - atn0_frac) ** 2

    reward = proximity_to_p if (
        (action == 0 and atn0_frac < env.p) or
        (action == 1 and atn0_frac >= env.p)
    ) else 0

    info = {}
    if terminal:
        info['score'] = proximity_to_p

    return reward, terminal, info

cdef void free_stochastic_env(CStochasticEnv* env):
    free(env)

cdef class CStochasticCy:
    cdef:
        CStochasticEnv* env
        float p
        int horizon

    def __init__(self, float p, int horizon):
        self.p = p
        self.horizon = horizon
        self.env = init_stochastic_env(p, horizon)

    def reset(self):
        reset(self.env)

    def step(self, int action):
        return step(self.env, action)

    def get_tick(self):
        return self.env.tick

    def get_action(self):
        return self.env.action

    def get_count(self):
        return self.env.count

    def __dealloc__(self):
        free_stochastic_env(self.env)
