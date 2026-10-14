#pragma once
#include "model.cuh"

// Exact scalar transmission. A follower owns no state; compilation aliases its
// qpos/dof to the source. Ratios multiply both motion and generalized reaction.
__host__ __device__ static inline float pf_joint_scale(const PfLink& link) {
    return link.mimic ? link.ratio : 1.0f;
}
__host__ __device__ static inline float pf_joint_position(const PfLink& link,
        const float* qpos) {
    return pf_joint_scale(link)*qpos[link.qpos] + (link.mimic ? link.offset : 0);
}
__host__ __device__ static inline float pf_joint_velocity(const PfLink& link,
        const float* qvel) {
    return pf_joint_scale(link)*qvel[link.dof];
}
