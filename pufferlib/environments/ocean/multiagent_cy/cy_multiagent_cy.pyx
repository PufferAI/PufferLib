import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t

cdef struct CMultiagentEnv:
    int num_agents
    int tick
    float* observation1  # Observation for agent 1
    float* observation2  # Observation for agent 2
    int32_t* actions     # Actions for both agents
    float* rewards       # Rewards for both agents
    float* view          # The view array that gets updated in step()

cdef CMultiagentEnv* init_c_multiagent_env(
    int num_agents, 
    float* observation1, 
    float* observation2, 
    int32_t* actions, 
    float* rewards, 
    float* view):
    
    cdef CMultiagentEnv* env = <CMultiagentEnv*> malloc(sizeof(CMultiagentEnv))
    env.num_agents = num_agents
    env.tick = 0
    env.observation1 = observation1
    env.observation2 = observation2
    env.actions = actions
    env.rewards = rewards
    env.view = view
    
    return env

cdef void reset(CMultiagentEnv* env):
    env.tick = 0
    # Reset observations and view
    env.observation1[0] = 0.0
    env.observation2[0] = 1.0
    for i in range(10):
        env.view[i] = 0.0

cdef void step(CMultiagentEnv* env):
    cdef int action1 = env.actions[0]  # Action for agent 1
    cdef int action2 = env.actions[1]  # Action for agent 2

    # Reset rewards before calculation
    env.rewards[0] = 0.0  # Reward for agent 1
    env.rewards[1] = 0.0  # Reward for agent 2

    # Assuming the view is a 2x5 grid flattened into a 1D array
    cdef int num_cols = 5  # 5 columns

    # Agent 1 should get reward for action 0
    if action1 == 0:
        env.view[0 * num_cols + 0] = 1.0  # Update correct column for Agent 1's action (top row)
        env.rewards[0] = 1.0  # Correct action, reward agent 1
    else:
        env.view[0 * num_cols + 1] = 1.0  # Update different column for Agent 1 (top row)
        env.rewards[0] = 0.0  # Incorrect action, no reward

    # Agent 2 should get reward for action 1
    if action2 == 1:
        env.view[1 * num_cols + 0] = 1.0  # Update correct column for Agent 2's action (bottom row)
        env.rewards[1] = 1.0  # Correct action, reward agent 2
    else:
        env.view[1 * num_cols + 1] = 1.0  # Update different column for Agent 2 (bottom row)
        env.rewards[1] = 0.0  # Incorrect action, no reward

    # Increment the tick counter
    env.tick += 1


cdef void free_c_multiagent_env(CMultiagentEnv* env):
    free(env)

cdef class CMultiagentCy:
    cdef:
        CMultiagentEnv* env
        int num_agents

    def __init__(self, 
                int num_agents,
                cnp.ndarray[cnp.float32_t, ndim=1] observation1,
                cnp.ndarray[cnp.float32_t, ndim=1] observation2,
                cnp.ndarray[cnp.int32_t, ndim=1] actions,
                cnp.ndarray[cnp.float32_t, ndim=1] rewards,
                cnp.ndarray[cnp.float32_t, ndim=2] view):
        
        self.num_agents = num_agents
        
        self.env = init_c_multiagent_env(
            num_agents, 
            <float*>observation1.data, 
            <float*>observation2.data, 
            <int32_t*>actions.data, 
            <float*>rewards.data,
            <float*>view.data
        )

    def reset(self):
        reset(self.env)

    def step(self):
        step(self.env)
    
    def get_tick(self):
        return self.env.tick

    def __dealloc__(self):
        free_c_multiagent_env(self.env)
