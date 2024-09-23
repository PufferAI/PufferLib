import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
import time

cdef struct CMemoryEnv:
    int mem_length
    int mem_delay
    int horizon
    int tick
    float* solution
    float* submission
    float* rewards
    int32_t* actions

cdef CMemoryEnv* init_c_memory_env(int mem_length, 
                                    int mem_delay, 
                                   float* solution, 
                                   float* submission, 
                                   float* rewards, 
                                   int32_t* actions
    ):
    cdef CMemoryEnv* env = <CMemoryEnv*> malloc(sizeof(CMemoryEnv))
    env.mem_length = mem_length
    env.mem_delay = mem_delay
    env.horizon = 2 * mem_length + mem_delay
    env.tick = 1
    env.solution = solution
    env.submission = submission
    env.rewards = rewards
    env.actions = actions

    return env

cdef void reset(CMemoryEnv* env, int seed=-1):
    env.tick = 1

    if seed != -1:
        np.random.seed(seed)
    else:
        np.random.seed(np.random.randint(0, 10000))
        
    for i in range(env.horizon):
        env.solution[i] = np.random.randint(0, 2)
    
    for i in range(env.mem_length + env.mem_delay):
        env.solution[env.horizon - i - 1] = -1

    for i in range(env.horizon):
        env.submission[i] = -1

cdef void step(CMemoryEnv* env):
    cdef int action = env.actions[0]
    env.rewards[0] = 0.0
    cdef float ob = 0.0

    if env.tick < env.mem_length:
        ob = env.solution[env.tick]
        env.rewards[0] = float(action == 0)

    elif env.tick >= env.mem_length + env.mem_delay:
        idx = env.tick - env.mem_length - env.mem_delay
        sol = env.solution[idx]
        env.rewards[0] = float(action == sol)
        env.submission[env.tick] = action

    env.tick += 1

cdef int is_done(CMemoryEnv* env):
    return env.tick == env.horizon

cdef int check_solution(CMemoryEnv* env):
    cdef int i
    for i in range(env.mem_length):
        if env.solution[i] != env.submission[env.horizon - env.mem_length + i]:
            return 0
    return 1

cdef void free_c_memory_env(CMemoryEnv* env):
    free(env)

cdef class CMemoryCy:
    cdef:
        CMemoryEnv* env

    def __init__(self, 
                int mem_length, int mem_delay, 
                 cnp.ndarray[cnp.float32_t, ndim=2] solution,
                 cnp.ndarray[cnp.float32_t, ndim=2] submission,
                 cnp.ndarray[cnp.float32_t, ndim=2] rewards, 
                 cnp.ndarray[cnp.int32_t, ndim=2] actions):
        
        self.env = init_c_memory_env(
            mem_length, mem_delay,
            <float*>solution.data, <float*>submission.data,
            <float*>rewards.data, <int32_t*>actions.data)

    def reset(self, int seed=-1):
        reset(self.env, seed)

    def step(self):
        step(self.env)

    def is_done(self):
        return is_done(self.env)

    def check_solution(self):
        return check_solution(self.env)

    def get_tick(self):
        return self.env.tick

    def __dealloc__(self):
        free_c_memory_env(self.env)
