# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.string cimport memset, memcpy
from libc.math cimport fmaxf, fminf, expf
import cython

def rewards_and_masks(
        float[:, :] reward_block,
        float[:, :] reward_mask,
        float[:, :] values_mean,
        float[:, :] values_std,
        float[:, :] buf,
        float[:] dones,
        float[:] rewards,
        float[:] advantages,
        int[:] bounds,
        int horizon,
        float vstd_min,
        float vstd_max
    ):

    cdef int num_steps = len(rewards)
    memset(&reward_mask[0, 0], 0, num_steps * horizon * sizeof(float))
    memset(&advantages[0], 0, num_steps * sizeof(float))

    cdef:
        int i, j, k, t
        float r

    for i in range(num_steps):
        k = 0
        for j in range(horizon):
            t = i + j

            if dones[t]:
                break

            if t >= num_steps - 1:
                break

            k += 1

            reward_block[i, j] = rewards[t+1]
            reward_mask[i, j] = 1.0

            # Store value std in buffer
            vstd = values_std[i, j]
            buf[i, j] = vstd

        bounds[i] = k

    cdef float delta = vstd_min - vstd_max
    if delta == 0:
        for i in range(num_steps):
            k = bounds[i]
            for j in range(k):
                advantages[i] += (reward_block[i, j] - values_mean[i, j])

        return

    cdef float adv_scale, adv_sum
    for i in range(num_steps):
        k = bounds[i]
        adv_sum = 0
        for j in range(k):
            adv_scale = (vstd_max - buf[i, j]) / delta
            adv_scale = adv_scale if adv_scale > 0.05 else 0.05
            adv_scale = adv_scale if adv_scale < 1 else 1
            buf[i, j] = adv_scale
            adv_sum += adv_scale

        for j in range(k):
            advantages[i] += buf[i, j]/adv_sum * (reward_block[i, j] - values_mean[i, j])
 
def fast_rewards_and_masks(float[:, :] reward_block, float[:, :] reward_mask,
        float[:] dones, float[:] rewards, int horizon):
    cdef int num_steps = len(rewards)
    cdef int i, h 
    for i in range(num_steps):
        h = horizon
        if i + h >= num_steps:
            h = num_steps - i - 1

        memcpy(&reward_block[i, 0], &rewards[i+1], h * sizeof(float))

def compute_gae(cnp.ndarray dones, float[:, :] values,
        float[:, :] rewards, float gamma, float gae_lambda):
    '''Fast Cython implementation of Generalized Advantage Estimation (GAE)'''
    cdef:
        float[:, :] c_dones = dones
        int num_rows = dones.shape[0]
        int horizon = dones.shape[1]
        float lastgaelam = 0
        float nextnonterminal, delta
        int t, t_cur, t_next
        cnp.ndarray advantages = np.zeros((num_rows, horizon), dtype=np.float32)
        float[:, :] c_advantages = advantages

    for row in range(num_rows-1, -1, -1):
        lastgaelam = 0
        for t in range(horizon-2, -1, -1):
            t_next = t + 1
            nextnonterminal = 1.0 - c_dones[row, t_next]
            delta = rewards[row, t_next] + gamma*values[row, t_next]*nextnonterminal - values[row, t]
            lastgaelam = delta + gamma*gae_lambda*nextnonterminal * lastgaelam
            c_advantages[row, t] = lastgaelam

    return advantages
