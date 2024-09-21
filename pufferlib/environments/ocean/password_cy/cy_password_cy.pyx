import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
import random


cdef struct CPasswordEnv:
    int password_length
    int hard_fixed_seed
    int tick
    float* observation
    float* solution

cdef CPasswordEnv* init_c_password_env(int password_length, int hard_fixed_seed):
    cdef CPasswordEnv* env = <CPasswordEnv*> malloc(sizeof(CPasswordEnv))
    env.password_length = password_length
    env.hard_fixed_seed = hard_fixed_seed
    env.observation = <float*> malloc(password_length * sizeof(float))
    env.solution = <float*> malloc(password_length * sizeof(float))
    env.tick = 0

    random.seed(hard_fixed_seed)
    np.random.seed(hard_fixed_seed)

    for i in range(password_length):
        env.observation[i] = -1.0
        env.solution[i] = np.random.randint(0, 2)

    return env


cdef void reset(CPasswordEnv* env):
    random.seed(env.hard_fixed_seed)
    np.random.seed(env.hard_fixed_seed)
    
    for i in range(env.password_length):
        env.observation[i] = -1.0
        env.solution[i] = np.random.randint(0, 2)

    env.tick = 0


cdef tuple step(CPasswordEnv* env, int action):
    assert env.tick < env.password_length
    assert action in (0, 1)

    env.observation[env.tick] = action
    env.tick += 1

    reward = 0.0
    terminal = env.tick == env.password_length

    if terminal:
        correct = True
        for i in range(env.password_length):
            if env.observation[i] != env.solution[i]:
                correct = False
                break
        if correct:
            reward = 1.0

    return reward, terminal


cdef void free_c_password_env(CPasswordEnv* env):
    free(env.observation)
    free(env.solution)
    free(env)

cdef class CPasswordCy:
    cdef:
        CPasswordEnv* env
        int password_length
        int hard_fixed_seed

    def __init__(self, int password_length=5, int hard_fixed_seed=42):
        self.password_length = password_length
        self.hard_fixed_seed = hard_fixed_seed
        self.env = init_c_password_env(password_length, hard_fixed_seed)

    def reset(self):
        reset(self.env)
        return np.array([self.env.observation[i] for i in range(self.password_length)], dtype=np.float32)

    def step(self, int action):
        reward, terminal = step(self.env, action)
        return (np.array([self.env.observation[i] for i in range(self.password_length)], dtype=np.float32), 
                reward, terminal)

    def get_solution(self):
        return np.array([self.env.solution[i] for i in range(self.password_length)], dtype=np.float32)

    def get_observation(self):
        return np.array([self.env.observation[i] for i in range(self.password_length)], dtype=np.float32)

    def __dealloc__(self):
        free_c_password_env(self.env)
