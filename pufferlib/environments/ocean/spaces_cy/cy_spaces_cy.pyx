import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int8_t

cdef struct CSpacesCyEnv:
    float* image_observations
    int8_t* flat_observations
    unsigned int* actions
    float* rewards
    unsigned char* dones
    int* scores
    int num_agents
    float image_sign
    float flat_sign

cdef CSpacesCyEnv* init_c_spaces_cy(float* image_observations, 
                                int8_t* flat_observations,
                                  unsigned int* actions, 
                                  float* rewards, 
                                  unsigned char* dones, 
                                  int* scores, 
                                  int num_agents
                                  ):

    cdef CSpacesCyEnv* env = <CSpacesCyEnv*> malloc(sizeof(CSpacesCyEnv))
    
    env.image_observations = image_observations
    env.flat_observations = flat_observations
    env.actions = actions
    env.rewards = rewards
    env.dones = dones
    env.scores = scores
    env.num_agents = num_agents

    return env

cdef void reset(CSpacesCyEnv* env):
    cdef int i, j
    for i in range(5):
        for j in range(5):
            env.image_observations[i * 5 + j] = np.random.randn()
        env.flat_observations[i] = np.random.randint(-1, 2) 

    cdef float image_sum = 0
    cdef float flat_sum = 0
    for i in range(5):
        for j in range(5):
           image_sum += env.image_observations[i * 5 + j]
        flat_sum += env.flat_observations[i]
    
    env.image_sign = image_sum > 0
    env.flat_sign = flat_sum > 0

cdef void step(CSpacesCyEnv* env):
    env.rewards[0] = 0.0

    if env.actions[0] == env.image_sign:
        env.rewards[0] += 0.5

    if env.actions[1] == env.flat_sign:
        env.rewards[0] += 0.5

    # 1-step environment
    env.dones[0] = 1

cdef void free_c_spaces_cy(CSpacesCyEnv* env):
    free(env)

# Cython wrapper class
cdef class CSpacesCy:
    cdef:
        CSpacesCyEnv* env

    def __init__(self, 
                cnp.ndarray[cnp.float32_t, ndim=3] image_observations, 
                 cnp.ndarray[cnp.int8_t, ndim=2] flat_observations,
                 cnp.ndarray[cnp.uint32_t, ndim=2] actions,
                 cnp.ndarray[cnp.float32_t, ndim=2] rewards, 
                 cnp.ndarray[cnp.uint8_t, ndim=2] dones, 
                 cnp.ndarray[cnp.int32_t, ndim=2] scores, 
                 int num_agents
                 ):

        self.env = init_c_spaces_cy(<float*>image_observations.data,
                                    <int8_t*>flat_observations.data, 
                                   <unsigned int*>actions.data, 
                                   <float*>rewards.data, 
                                   <unsigned char*>dones.data,
                                   <int*>scores.data, 
                                   num_agents
                                   )
    
    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

    def __dealloc__(self):
        free_c_spaces_cy(self.env)
