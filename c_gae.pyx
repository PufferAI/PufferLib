# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp

def compute_gae(cnp.ndarray dones, cnp.ndarray values,
        cnp.ndarray rewards, float gamma, float gae_lambda):
    '''Fast Cython implementation of Generalized Advantage Estimation (GAE)'''
    cdef int num_steps = len(rewards)
    cdef cnp.ndarray advantages = np.zeros(num_steps, dtype=np.float32)
    cdef float[:] c_advantages = advantages
    cdef float[:] c_dones = dones
    cdef float[:] c_values = values
    cdef float[:] c_rewards = rewards

    cdef float lastgaelam = 0
    cdef float nextnonterminal, delta
    cdef int t, t_cur, t_next
    for t in range(num_steps-1):
        t_cur = num_steps - 2 - t
        t_next = num_steps - 1 - t
        nextnonterminal = 1.0 - c_dones[t_next]
        delta = c_rewards[t_next] + gamma * c_values[t_next] * nextnonterminal - c_values[t_cur]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        c_advantages[t_cur] = lastgaelam

    return advantages


