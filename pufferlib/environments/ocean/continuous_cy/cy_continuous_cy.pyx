import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
import random

cnp.import_array()

cdef struct CContinuousEnv:
    float* state  # Pointer to state array (pos_x, pos_y, vel_x, vel_y, target_x, target_y)
    int tick
    bint discretize

cdef CContinuousEnv* init_continuous_env(bint discretize):
    cdef CContinuousEnv* env = <CContinuousEnv*> malloc(sizeof(CContinuousEnv))
    env.state = <float*> malloc(6 * sizeof(float))
    env.tick = 0
    env.discretize = discretize
    return env

# Reset the environment
cdef void reset(CContinuousEnv* env):
    env.tick = 0
    env.state[0] = 2 * np.random.rand() - 1  # pos_x
    env.state[1] = 2 * np.random.rand() - 1  # pos_y
    env.state[2] = 0  # vel_x
    env.state[3] = 0  # vel_y
    env.state[4] = 2 * np.random.rand() - 1  # target_x
    env.state[5] = 2 * np.random.rand() - 1  # target_y

cdef tuple step(CContinuousEnv* env, object action):
    cdef float accel_x, accel_y
    cdef int action_int
    cdef cnp.ndarray[cnp.float32_t, ndim=1] action_array = action

    if env.discretize:
        action_int = <int>action
        if action_int == 0:
            accel_x, accel_y = -0.1, 0
        elif action_int == 1:
            accel_x, accel_y = 0.1, 0
        elif action_int == 2:
            accel_x, accel_y = 0, -0.1
        elif action_int == 3:
            accel_x, accel_y = 0, 0.1
    else:
        accel_x, accel_y = 0.1 * action_array[0], 0.1 * action_array[1]

    env.state[2] += accel_x  # vel_x
    env.state[3] += accel_y  # vel_y
    env.state[0] += env.state[2]  # pos_x
    env.state[1] += env.state[3]  # pos_y

    cdef float pos_x = env.state[0]
    cdef float pos_y = env.state[1]
    cdef float vel_x = env.state[2]
    cdef float vel_y = env.state[3]
    cdef float target_x = env.state[4]
    cdef float target_y = env.state[5]

    if pos_x < -1 or pos_x > 1 or pos_y < -1 or pos_y > 1:
        return (<float[:6]>env.state)[:], -1.0, True, False, {'score': 0}

    dist = np.sqrt((pos_x - target_x) ** 2 + (pos_y - target_y) ** 2)
    reward = 0.02 * (1 - dist)

    env.tick += 1
    done = dist < 0.1
    truncated = env.tick >= 100

    info = {}
    if done:
        reward = 5.0
        info['score'] = 1
    elif truncated:
        reward = 0.0
        info['score'] = 0

    return (<float[:6]>env.state)[:], reward, done, truncated, info

cdef void free_continuous_env(CContinuousEnv* env):
    free(env.state)
    free(env)

cdef class CContinuousCy:
    cdef:
        CContinuousEnv* env
        bint discretize

    def __init__(self, bint discretize):
        self.discretize = discretize
        self.env = init_continuous_env(discretize)

    def reset(self):
        reset(self.env)

    def step(self, object action):
        return step(self.env, action)

    def get_state(self):
        return (<float[:6]>self.env.state)[:]

    def get_tick(self):
        return self.env.tick

    def __dealloc__(self):
        free_continuous_env(self.env)
