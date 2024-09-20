import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t

cdef struct CBanditEnv:
    int num_actions
    int solution_idx
    float reward_scale
    float reward_noise
    int hard_fixed_seed
    float* rewards
    int32_t* actions

cdef CBanditEnv* init_c_bandit_env(int num_actions, 
                                float reward_scale, 
                                  float reward_noise, 
                                  int hard_fixed_seed,
                                  float* rewards, 
                                  int32_t* actions
    ):
    # Allocate memory for the environment
    cdef CBanditEnv* env = <CBanditEnv*> malloc(sizeof(CBanditEnv))
    env.num_actions = num_actions
    env.reward_scale = reward_scale
    env.reward_noise = reward_noise
    env.hard_fixed_seed = hard_fixed_seed
    env.rewards = rewards
    env.actions = actions

    # Set up the solution
    np.random.seed(hard_fixed_seed)
    env.solution_idx = np.random.randint(0, num_actions)
    
    return env

cdef void reset(CBanditEnv* env):
    np.random.seed(env.hard_fixed_seed)
    env.solution_idx = np.random.randint(0, env.num_actions)

cdef void step(CBanditEnv* env):
    cdef int action = env.actions[0]
    env.rewards[0] = 0.0

    if action == env.solution_idx:
        env.rewards[0] = 1.0

    if env.reward_noise != 0.0:
        env.rewards[0] += np.random.randn() * env.reward_scale

    env.rewards[0] *= env.reward_scale

cdef void free_c_bandit_env(CBanditEnv* env):
    free(env)

# Cython wrapper class
cdef class CBanditCy:
    cdef:
        CBanditEnv* env

    def __init__(self, 
                int num_actions, 
                float reward_scale, 
                 float reward_noise, 
                 int hard_fixed_seed, 
                 cnp.ndarray[cnp.float32_t, ndim=2] rewards, 
                 cnp.ndarray[cnp.int32_t, ndim=2] actions
        ):
        
        self.env = init_c_bandit_env(
            num_actions, reward_scale, reward_noise, hard_fixed_seed,
            <float*>rewards.data, <int32_t*>actions.data)

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)

    def get_solution_idx(self):
        return self.env.solution_idx

    def __dealloc__(self):
        free_c_bandit_env(self.env)
