#pragma once

#include "forward_dynamics.cuh"
#include "integrate.cuh"

#define PF_NAIVE_JOINT_ITERATIONS 8

__device__ static inline void pf_naive_angular_pair(
        PfBody* parent, PfBody* child, PfVec3 axis) {
    PfVec3 reference = fabsf(axis.y) < 0.9f ? pf_v3(0.0f, 1.0f, 0.0f)
        : pf_v3(1.0f, 0.0f, 0.0f);
    PfVec3 tangent_1 = pf_normalize_or(pf_cross(reference, axis),
        pf_v3(1.0f, 0.0f, 0.0f));
    PfVec3 tangent_2 = pf_normalize_or(pf_cross(axis, tangent_1),
        pf_v3(0.0f, 1.0f, 0.0f));
    PfVec3 directions[2] = {tangent_1, tangent_2};
    for (int row = 0; row < 2; ++row) {
        PfVec3 direction = directions[row];
        float relative = pf_dot(pf_sub(child->angular_velocity,
            parent->angular_velocity), direction);
        float effective = 0.0f;
        if (pf_mode_is_dynamic(parent->mode)) effective += pf_dot(direction,
            pf_inverse_inertia_world(parent, direction));
        if (pf_mode_is_dynamic(child->mode)) effective += pf_dot(direction,
            pf_inverse_inertia_world(child, direction));
        if (effective <= 1.0e-8f) continue;
        /* Child takes the impulse sized from child-minus-parent spin, parent
         * takes the opposite one. Applying it the other way round grows the
         * very component the constraint removes. */
        float impulse = -relative / effective;
        if (pf_mode_is_dynamic(child->mode)) child->angular_velocity =
            pf_add(child->angular_velocity, pf_inverse_inertia_world(child,
                pf_scale(direction, impulse)));
        if (pf_mode_is_dynamic(parent->mode)) parent->angular_velocity =
            pf_add(parent->angular_velocity, pf_inverse_inertia_world(parent,
                pf_scale(direction, -impulse)));
    }
}

__device__ static inline bool pf_naive_maximal_chain_step(
        PfJointWorld world, float gravity, float dt, int iterations) {
    if (world.joints == NULL || world.bodies == NULL
            || world.joint_count <= 0 || world.joint_count > 9
            || world.body_count != world.joint_count + 1
            || !pf_number(gravity) || !pf_number(dt) || dt <= 0.0f
            || iterations <= 0 || !pf_joint_forward_kinematics(world)) return false;
    for (int index = 0; index < world.joint_count; ++index) {
        const PfJoint* joint = &world.joints[index];
        if (joint->parent_body != index || joint->child_body != index + 1
                || !pf_joint_valid(joint, world.body_count)) return false;
    }
    for (int body = 0; body < world.body_count; ++body) {
        PfBody* current = &world.bodies[body];
        if (pf_mode_is_dynamic(current->mode)) current->linear_velocity.y += gravity * dt;
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int index = 0; index < world.joint_count; ++index) {
            const PfJoint* joint = &world.joints[index];
            PfBody* child = &world.bodies[joint->child_body];
            PfBody* parent = &world.bodies[joint->parent_body];
            PfVec3 parent_anchor = pf_add(parent->position,
                pf_quat_rotate(parent->rotation, joint->parent_anchor));
            PfVec3 child_anchor = pf_add(child->position,
                pf_quat_rotate(child->rotation, joint->child_anchor));
            PfVec3 directions[3] = {pf_v3(1.0f, 0.0f, 0.0f),
                pf_v3(0.0f, 1.0f, 0.0f), pf_v3(0.0f, 0.0f, 1.0f)};
            for (int row = 0; row < 3; ++row) {
                PfVec3 direction = directions[row];
                float relative = pf_dot(pf_sub(pf_point_velocity(parent, parent_anchor),
                    pf_point_velocity(child, child_anchor)), direction);
                float effective = pf_effective_mass(parent, child,
                    parent_anchor, child_anchor, direction);
                if (effective <= 1.0e-8f) continue;
                float impulse = -relative / effective;
                pf_apply_impulse(parent, parent_anchor, pf_scale(direction, impulse));
                pf_apply_impulse(child, child_anchor, pf_scale(direction, -impulse));
            }
            pf_naive_angular_pair(parent, child,
                pf_quat_rotate(parent->rotation, joint->axis));
        }
    }
    for (int body = 0; body < world.body_count; ++body) {
        PfBody* current = &world.bodies[body];
        if (!pf_mode_is_dynamic(current->mode)) continue;
        current->position = pf_add(current->position, pf_scale(current->linear_velocity, dt));
        current->rotation = pf_quat_normalize(pf_quat_multiply(
            pf_orientation_delta(current->angular_velocity, dt), current->rotation));
    }
    for (int index = 0; index < world.joint_count; ++index) {
        PfJoint* joint = &world.joints[index];
        PfBody* child = &world.bodies[joint->child_body];
        PfBody* parent = &world.bodies[joint->parent_body];
        PfVec3 relative = pf_sub(child->angular_velocity, parent->angular_velocity);
        joint->angular_velocity = pf_dot(relative,
            pf_joint_world_axis(*joint, world.bodies));
        joint->angle = fmaxf(joint->lower_limit, fminf(joint->upper_limit,
            pf_joint_angle(*joint, world.bodies)));
    }
    return true;
}

__global__ static void pf_naive_maximal_chain_kernel(
        PfJoint* joints, PfBody* bodies, int env_count, int joint_count,
        int joint_stride, int body_stride, float gravity, float dt,
        int iterations) {
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) return;
    PfJointWorld world = {};
    world.joints = joints + (size_t)env * joint_stride;
    world.bodies = bodies + (size_t)env * body_stride;
    world.joint_count = joint_count;
    world.body_count = joint_count + 1;
    world.joint_capacity = joint_stride;
    world.body_capacity = body_stride;
    (void)pf_naive_maximal_chain_step(world, gravity, dt, iterations);
}
