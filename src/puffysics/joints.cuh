#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <string.h>

#include "batch.cuh"
#include "contact_solver.cuh"

#define PF_JOINT_VELOCITY_ITERATIONS 16
#define PF_JOINT_POSITION_ITERATIONS 16
#define PF_JOINT_POSITION_SLOP 0.0005f
#define PF_JOINT_POSITION_PERCENT 1.0f
#define PF_JOINT_MAX_POSITION_CORRECTION 0.1f
#define PF_JOINT_LIMIT_SLOP 0.0005f

#define PF_REDUCED_BASE_DOF 6
#define PF_REDUCED_MAX_DOF 64
#define PF_REDUCED_MAX_BODIES 10
static inline void pf_joint_batch_destroy(PfJointBatch* batch);

static inline bool pf_reduced_workspace_floats(int joint_capacity,
        size_t* result) {
    if (result == NULL || joint_capacity <= 0) return false;
    size_t dof = (size_t)joint_capacity + PF_REDUCED_BASE_DOF;
    size_t body = (size_t)joint_capacity + 1;
    size_t matrix = 0;
    size_t body_arrays = 0;
    size_t joint_arrays = 0;
    size_t total = 0;
    if (!pf_size_mul(dof, dof, &matrix)
            || !pf_size_mul(body, (size_t)20, &body_arrays)
            || !pf_size_mul((size_t)joint_capacity, (size_t)6, &joint_arrays)
            || matrix > (size_t)-1 - 2 * dof
            || matrix + 2 * dof > (size_t)-1 - body_arrays
            || matrix + 2 * dof + body_arrays > (size_t)-1 - joint_arrays) {
        return false;
    }
    total = matrix + 2 * dof + body_arrays + joint_arrays;
    *result = total;
    return total <= 0x7fffffffu;
}

__host__ __device__ static inline bool pf_joint_revolute(PfJoint* out,
        int parent_body, int child_body, PfVec3 axis,
        PfVec3 parent_anchor, PfVec3 child_anchor,
        float lower_limit, float upper_limit) {
    if (out == NULL || parent_body < -1 || child_body < 0
            || parent_body >= child_body || !pf_vec_valid(axis)
            || !pf_vec_valid(parent_anchor) || !pf_vec_valid(child_anchor)
            || !pf_number(lower_limit) || !pf_number(upper_limit)
            || lower_limit > upper_limit
            || pf_length_squared(axis) <= 1.0e-20f) {
        return false;
    }
    PfJoint joint = {};
    joint.type = PF_JOINT_REVOLUTE;
    joint.parent_body = parent_body;
    joint.child_body = child_body;
    joint.axis = pf_normalize_or(axis, pf_v3(0.0f, 0.0f, 1.0f));
    joint.parent_anchor = parent_anchor;
    joint.child_anchor = child_anchor;
    joint.lower_limit = lower_limit;
    joint.upper_limit = upper_limit;
    *out = joint;
    return true;
}

__host__ __device__ static inline bool pf_joint_set_motor(PfJoint* joint,
        float target, float max_torque, float stiffness, float damping) {
    if (joint == NULL || !pf_number(target) || !pf_number(max_torque)
            || max_torque < 0.0f || !pf_number(stiffness) || stiffness < 0.0f
            || !pf_number(damping) || damping < 0.0f) {
        return false;
    }
    PfJoint updated = *joint;
    updated.motor_target = target;
    updated.motor_max_torque = max_torque;
    updated.motor_stiffness = stiffness;
    updated.motor_damping = damping;
    *joint = updated;
    return true;
}

__host__ __device__ static inline bool pf_joint_set_damping(PfJoint* joint,
        float armature, float damping) {
    if (joint == NULL || !pf_number(armature) || armature < 0.0f
            || !pf_number(damping) || damping < 0.0f) {
        return false;
    }
    PfJoint updated = *joint;
    updated.armature = armature;
    updated.damping = damping;
    *joint = updated;
    return true;
}

/* Body-index contract, checked before any bodies[] dereference. -1 is the
 * world root; every other parent must precede its child. */
__host__ __device__ static inline bool pf_joint_index_ordered(
        const PfJoint* joint) {
    return joint != NULL && joint->parent_body >= -1 && joint->child_body >= 0
        && joint->parent_body < joint->child_body;
}

__host__ __device__ static inline bool pf_joint_index_valid(
        const PfJoint* joint, int body_count) {
    return pf_joint_index_ordered(joint) && joint->child_body < body_count;
}

__host__ __device__ static inline bool pf_joint_valid(
        const PfJoint* joint, int body_count) {
    if (joint == NULL || joint->type != PF_JOINT_REVOLUTE
            || !pf_joint_index_valid(joint, body_count)
            || !pf_vec_valid(joint->axis)
            || pf_length_squared(joint->axis) <= 1.0e-12f
            || !pf_vec_valid(joint->parent_anchor)
            || !pf_vec_valid(joint->child_anchor)
            || !pf_number(joint->angle) || !pf_number(joint->angular_velocity)
            || !pf_number(joint->lower_limit) || !pf_number(joint->upper_limit)
            || joint->lower_limit > joint->upper_limit
            || !pf_number(joint->motor_target)
            || !pf_number(joint->motor_max_torque)
            || joint->motor_max_torque < 0.0f
            || !pf_number(joint->motor_stiffness)
            || joint->motor_stiffness < 0.0f
            || !pf_number(joint->motor_damping)
            || joint->motor_damping < 0.0f
            || !pf_number(joint->armature) || joint->armature < 0.0f
            || !pf_number(joint->damping) || joint->damping < 0.0f) {
        return false;
    }
    PfVec3 unit = pf_normalize_or(joint->axis, pf_v3(0.0f, 0.0f, 1.0f));
    return pf_length_squared(pf_sub(joint->axis, unit)) <= 1.0e-6f;
}

static inline bool pf_joint_batch_create(PfJointBatch* batch,
        int env_count, int joint_capacity) {
    if (batch == NULL) {
        return false;
    }
    memset(batch, 0, sizeof(*batch));
    if (env_count <= 0 || joint_capacity <= 0) {
        return false;
    }
    size_t joint_count = 0;
    size_t joint_bytes = 0;
    if (!pf_size_mul((size_t)env_count, (size_t)joint_capacity, &joint_count)
            || joint_count > 0x7fffffffu
            || !pf_size_mul(joint_count, sizeof(PfJoint), &joint_bytes)) {
        return false;
    }
    size_t scratch_stride = 0;
    if (!pf_reduced_workspace_floats(joint_capacity, &scratch_stride)) {
        return false;
    }
    size_t scratch_count = 0;
    size_t scratch_bytes = 0;
    if (!pf_size_mul((size_t)env_count, scratch_stride, &scratch_count)
            || !pf_size_mul(scratch_count, sizeof(float), &scratch_bytes)) {
        return false;
    }
    if (cudaMalloc((void**)&batch->joints, joint_bytes) != cudaSuccess) {
        batch->joints = NULL;
        return false;
    }
    if (cudaMalloc((void**)&batch->scratch, scratch_bytes) != cudaSuccess) {
        batch->scratch = NULL;
        pf_joint_batch_destroy(batch);
        return false;
    }
    batch->env_count = env_count;
    batch->joint_capacity = joint_capacity;
    batch->joint_stride = (size_t)joint_capacity;
    batch->scratch_stride = scratch_stride;
    if (cudaMemset(batch->joints, 0, joint_bytes) != cudaSuccess) {
        pf_joint_batch_destroy(batch);
        return false;
    }
    return true;
}

static inline void pf_joint_batch_destroy(PfJointBatch* batch) {
    if (batch == NULL) {
        return;
    }
    if (batch->scratch != NULL) {
        cudaFree(batch->scratch);
    }
    if (batch->joints != NULL) {
        cudaFree(batch->joints);
    }
    memset(batch, 0, sizeof(*batch));
}

__device__ static inline bool pf_joint_world(const PfJointBatch& batch,
        PfBody* bodies, int env_index, int joint_count, int body_count,
        PfJointWorld* out) {
    if (out == NULL || env_index < 0 || env_index >= batch.env_count
            || joint_count <= 0 || joint_count > batch.joint_capacity
            || body_count < joint_count || bodies == NULL) {
        return false;
    }
    out->joints = batch.joints + (size_t)env_index * batch.joint_stride;
    /* `bodies` is the per-environment base pointer already offset by
     * pf_world (PfWorld::bodies), so the environment offset is applied
     * there and must NOT be applied a second time here. Every caller in
     * the tree passes that already-offset pointer together with the same
     * env_index; offsetting again would read environment 2*i and, past
     * the last half of the batch, write out of bounds. PfJointBatch
     * carries no body_stride, so the true stride is not even knowable
     * here - and scratch, four lines below, is offset exactly once. */
    out->bodies = bodies;
    out->joint_count = joint_count;
    out->body_count = body_count;
    out->joint_capacity = batch.joint_capacity;
    out->body_capacity = body_count;
    out->env_index = env_index;
    out->scratch = batch.scratch == NULL ? NULL
        : batch.scratch + (size_t)env_index * batch.scratch_stride;
    out->scratch_capacity = (int)batch.scratch_stride;
    return true;
}

__device__ static inline bool pf_joint_add_revolute(PfJointWorld* world,
        int parent_body, int child_body, PfVec3 axis,
        PfVec3 parent_anchor, PfVec3 child_anchor,
        float lower_limit, float upper_limit) {
    if (world == NULL || world->joints == NULL || world->bodies == NULL
            || world->joint_count < 0
            || world->joint_count >= world->joint_capacity
            || parent_body < -1 || child_body < 0
            || parent_body >= child_body || child_body >= world->body_count
            || !pf_mode_is_dynamic(world->bodies[child_body].mode)) {
        return false;
    }
    PfJoint joint;
    if (!pf_joint_revolute(&joint, parent_body, child_body, axis,
            parent_anchor, child_anchor, lower_limit, upper_limit)) {
        return false;
    }
    world->joints[world->joint_count++] = joint;
    return true;
}

__device__ static inline bool pf_joint_constraints_valid(
        const PfJoint* joints, int joint_count, const PfBody* bodies,
        int body_count) {
    if (joints == NULL || bodies == NULL || joint_count <= 0
            || body_count < joint_count) {
        return false;
    }
    for (int index = 0; index < joint_count; ++index) {
        if (!pf_joint_valid(&joints[index], body_count)
                || joints[index].parent_body >= joints[index].child_body) {
            return false;
        }
    }
    return true;
}

__device__ static inline PfVec3 pf_joint_world_axis(
        const PfJoint& joint, const PfBody* bodies) {
    PfVec3 axis = joint.parent_body < 0 ? joint.axis
        : pf_quat_rotate(bodies[joint.parent_body].rotation, joint.axis);
    return pf_normalize_or(axis, pf_v3(0.0f, 0.0f, 1.0f));
}

__device__ static inline void pf_joint_basis(PfVec3 axis,
        PfVec3* tangent_1, PfVec3* tangent_2) {
    PfVec3 reference = fabsf(axis.y) < 0.9f ? pf_v3(0.0f, 1.0f, 0.0f)
        : pf_v3(1.0f, 0.0f, 0.0f);
    *tangent_1 = pf_normalize_or(pf_cross(reference, axis),
        pf_v3(1.0f, 0.0f, 0.0f));
    *tangent_2 = pf_normalize_or(pf_cross(axis, *tangent_1),
        pf_v3(0.0f, 1.0f, 0.0f));
}

__device__ static inline float pf_joint_angle(const PfJoint& joint,
        const PfBody* bodies) {
    PfQuat relative = joint.parent_body < 0 ? bodies[joint.child_body].rotation
        : pf_quat_multiply(pf_quat_conjugate(
            bodies[joint.parent_body].rotation), bodies[joint.child_body].rotation);
    PfVec3 imaginary = pf_v3(relative.x, relative.y, relative.z);
    /* 2*atan2f is wrapped, so a joint that has turned past the origin
     * reads a small angle again. Every limit comparison downstream reads
     * this value, so once it aliases past a stop the limit row stops
     * being reachable and the joint spins up with nothing to stop it.
     *
     * Unwrap against the stored accumulator instead: the correction is
     * the true signed rotation since the last pf_joint_refresh_state,
     * which lands in (-pi, pi]. pf_joint_refresh_state writes the result
     * straight back to joint->angle, so the accumulator advances
     * continuously and is never re-wrapped - the fold-back that makes
     * this safe exists and is the assignment at the end of that
     * function. Correct while one refresh interval turns the joint by
     * less than pi; past that rintf picks the wrong branch, which is the
     * runaway this removes rather than a regime to support. */
    return 2.0f * atan2f(pf_dot(joint.axis, imaginary), relative.w);
}

__device__ static inline float pf_joint_point_mass(
        const PfBody* parent, const PfBody* child, PfVec3 parent_anchor,
        PfVec3 child_anchor, PfVec3 direction) {
    float mass = 0.0f;
    if (parent != NULL && pf_mode_is_dynamic(parent->mode)) {
        PfVec3 lever = pf_cross(pf_sub(parent_anchor, parent->position), direction);
        mass += parent->inverse_mass
            + pf_dot(lever, pf_inverse_inertia_world(parent, lever));
    }
    if (pf_mode_is_dynamic(child->mode)) {
        PfVec3 lever = pf_cross(pf_sub(child_anchor, child->position), direction);
        mass += child->inverse_mass
            + pf_dot(lever, pf_inverse_inertia_world(child, lever));
    }
    return mass;
}

__device__ static inline float pf_joint_point_cross_mass(
        const PfBody* parent, const PfBody* child, PfVec3 parent_anchor,
        PfVec3 child_anchor, PfVec3 direction_1, PfVec3 direction_2) {
    float mass = 0.0f;
    if (parent != NULL && pf_mode_is_dynamic(parent->mode)) {
        PfVec3 lever_1 = pf_cross(pf_sub(parent_anchor, parent->position), direction_1);
        PfVec3 lever_2 = pf_cross(pf_sub(parent_anchor, parent->position), direction_2);
        mass += pf_dot(direction_1, direction_2) * parent->inverse_mass
            + pf_dot(lever_1, pf_inverse_inertia_world(parent, lever_2));
    }
    if (pf_mode_is_dynamic(child->mode)) {
        PfVec3 lever_1 = pf_cross(pf_sub(child_anchor, child->position), direction_1);
        PfVec3 lever_2 = pf_cross(pf_sub(child_anchor, child->position), direction_2);
        mass += pf_dot(direction_1, direction_2) * child->inverse_mass
            + pf_dot(lever_1, pf_inverse_inertia_world(child, lever_2));
    }
    return mass;
}

/* ponytail: armature is a reduced-solver generalised inertia only. The
 * maximal path applies angular impulses straight to the two bodies and holds
 * no rotor state, so there is nothing to reflect; a 1/armature term here
 * softened the forbidden tangential directions instead. Add a rotor
 * accumulator to PfJoint when the maximal solver needs reflected inertia. */
__device__ static inline float pf_joint_angular_mass(
        const PfBody* parent, const PfBody* child, PfVec3 direction) {
    float mass = 0.0f;
    if (parent != NULL && pf_mode_is_dynamic(parent->mode)) {
        mass += pf_dot(direction, pf_inverse_inertia_world(parent, direction));
    }
    if (pf_mode_is_dynamic(child->mode)) {
        mass += pf_dot(direction, pf_inverse_inertia_world(child, direction));
    }
    return mass;
}

__device__ static inline void pf_joint_angular_pair(
        const PfBody* parent, const PfBody* child,
        PfVec3 tangent_1, PfVec3 tangent_2, PfVec3 relative,
        float* impulse_1, float* impulse_2) {
    PfVec3 a1 = pf_inverse_inertia_world(child, tangent_1);
    PfVec3 a2 = pf_inverse_inertia_world(child, tangent_2);
    if (parent != NULL && pf_mode_is_dynamic(parent->mode)) {
        a1 = pf_add(a1, pf_inverse_inertia_world(parent, tangent_1));
        a2 = pf_add(a2, pf_inverse_inertia_world(parent, tangent_2));
    }
    float m11 = pf_dot(tangent_1, a1);
    float m12 = pf_dot(tangent_1, a2);
    float m22 = pf_dot(tangent_2, a2);
    float determinant = m11 * m22 - m12 * m12;
    if (determinant <= 1.0e-12f) {
        *impulse_1 = 0.0f;
        *impulse_2 = 0.0f;
        return;
    }
    float rhs1 = -pf_dot(relative, tangent_1);
    float rhs2 = -pf_dot(relative, tangent_2);
    *impulse_1 = (rhs1 * m22 - rhs2 * m12) / determinant;
    *impulse_2 = (rhs2 * m11 - rhs1 * m12) / determinant;
}

__device__ static inline void pf_joint_apply_angular_impulse(PfBody* body,
        PfVec3 direction, float impulse) {
    if (pf_mode_is_dynamic(body->mode)) body->angular_velocity = pf_add(
        body->angular_velocity, pf_inverse_inertia_world(body,
            pf_scale(direction, impulse)));
}

__device__ static inline void pf_joint_apply_motors(PfJoint* joints,
        int joint_count, PfBody* bodies) {
    for (int index = 0; index < joint_count; ++index) {
        PfJoint* joint = &joints[index];
        if (!pf_joint_index_ordered(joint)) continue;
        PfBody* parent = joint->parent_body < 0 ? NULL
            : &bodies[joint->parent_body];
        PfBody* child = &bodies[joint->child_body];
        PfVec3 axis = pf_joint_world_axis(*joint, bodies);
        float angle = pf_joint_angle(*joint, bodies);
        PfVec3 parent_spin = parent == NULL ? pf_v3(0.0f, 0.0f, 0.0f)
            : parent->angular_velocity;
        float speed = pf_dot(pf_sub(child->angular_velocity, parent_spin), axis);
        /* The motor limit bounds the ACTIVE term only. Passive joint damping
         * is a separate load, so a zero motor torque cannot switch it off. */
        float motor = fmaxf(-joint->motor_max_torque,
            fminf(joint->motor_max_torque, joint->motor_stiffness
                * (joint->motor_target - angle)
                - joint->motor_damping * speed));
        float torque = motor - joint->damping * speed;
        if (pf_mode_is_dynamic(child->mode)) child->torque = pf_add(
            child->torque, pf_scale(axis, torque));
        if (parent != NULL && pf_mode_is_dynamic(parent->mode)) parent->torque = pf_add(
            parent->torque, pf_scale(axis, -torque));
    }
}

__device__ static inline void pf_joint_solve_velocity_once(
        PfJoint* joints, int joint_count, PfBody* bodies) {
    {
        for (int index = 0; index < joint_count; ++index) {
            PfJoint* joint = &joints[index];
            if (!pf_joint_index_ordered(joint)) continue;
            PfBody* parent = joint->parent_body < 0 ? NULL
                : &bodies[joint->parent_body];
            PfBody* child = &bodies[joint->child_body];
            PfVec3 parent_anchor = parent == NULL ? joint->parent_anchor
                : pf_add(parent->position, pf_quat_rotate(parent->rotation,
                    joint->parent_anchor));
            PfVec3 child_anchor = pf_add(child->position,
                pf_quat_rotate(child->rotation, joint->child_anchor));
            PfVec3 parent_velocity = parent == NULL ? pf_v3(0.0f, 0.0f, 0.0f)
                : pf_point_velocity(parent, parent_anchor);
            PfVec3 relative = pf_sub(parent_velocity,
                pf_point_velocity(child, child_anchor));
            PfVec3 directions[3] = {pf_v3(1.0f, 0.0f, 0.0f),
                pf_v3(0.0f, 1.0f, 0.0f), pf_v3(0.0f, 0.0f, 1.0f)};
            float matrix[3][3];
            float rhs[3];
            for (int row = 0; row < 3; ++row) {
                rhs[row] = -pf_dot(relative, directions[row]);
                for (int column = 0; column < 3; ++column) {
                    matrix[row][column] = pf_joint_point_cross_mass(parent, child,
                        parent_anchor, child_anchor, directions[row], directions[column]);
                }
            }
            for (int pivot = 0; pivot < 3; ++pivot) {
                int best = pivot;
                for (int row = pivot + 1; row < 3; ++row)
                    if (fabsf(matrix[row][pivot]) > fabsf(matrix[best][pivot])) best = row;
                if (best != pivot) {
                    for (int column = pivot; column < 3; ++column) {
                        float swap = matrix[pivot][column];
                        matrix[pivot][column] = matrix[best][column];
                        matrix[best][column] = swap;
                    }
                    float swap = rhs[pivot]; rhs[pivot] = rhs[best]; rhs[best] = swap;
                }
                if (fabsf(matrix[pivot][pivot]) <= 1.0e-8f) continue;
                for (int row = pivot + 1; row < 3; ++row) {
                    float factor = matrix[row][pivot] / matrix[pivot][pivot];
                    for (int column = pivot; column < 3; ++column)
                        matrix[row][column] -= factor * matrix[pivot][column];
                    rhs[row] -= factor * rhs[pivot];
                }
            }
            float impulse[3] = {0.0f, 0.0f, 0.0f};
            for (int row = 2; row >= 0; --row) {
                float value = rhs[row];
                for (int column = row + 1; column < 3; ++column)
                    value -= matrix[row][column] * impulse[column];
                if (fabsf(matrix[row][row]) > 1.0e-8f)
                    impulse[row] = value / matrix[row][row];
            }
            for (int row = 0; row < 3; ++row) {
                if (parent != NULL) pf_apply_impulse(parent, parent_anchor,
                    pf_scale(directions[row], impulse[row]));
                pf_apply_impulse(child, child_anchor,
                    pf_scale(directions[row], -impulse[row]));
            }
            PfVec3 axis = pf_joint_world_axis(*joint, bodies);
            PfVec3 tangent_1, tangent_2;
            pf_joint_basis(axis, &tangent_1, &tangent_2);
            PfVec3 parent_spin = parent == NULL ? pf_v3(0.0f, 0.0f, 0.0f)
                : parent->angular_velocity;
            PfVec3 relative_spin = pf_sub(child->angular_velocity, parent_spin);
            float impulse_1, impulse_2;
            pf_joint_angular_pair(parent, child, tangent_1, tangent_2,
                relative_spin, &impulse_1, &impulse_2);
            pf_joint_apply_angular_impulse(child, tangent_1, impulse_1);
            pf_joint_apply_angular_impulse(child, tangent_2, impulse_2);
            if (parent != NULL) {
                pf_joint_apply_angular_impulse(parent, tangent_1, -impulse_1);
                pf_joint_apply_angular_impulse(parent, tangent_2, -impulse_2);
            }
            float angle = pf_joint_angle(*joint, bodies);
            parent_spin = parent == NULL ? pf_v3(0.0f, 0.0f, 0.0f)
                : parent->angular_velocity;
            float spin = pf_dot(pf_sub(child->angular_velocity, parent_spin), axis);
            float mass = pf_joint_angular_mass(parent, child, axis);
            if (mass <= 1.0e-8f) continue;
            if (angle < joint->lower_limit + PF_JOINT_LIMIT_SLOP && spin < 0.0f) {
                float impulse = -spin / mass;
                pf_joint_apply_angular_impulse(child, axis, impulse);
                if (parent != NULL) pf_joint_apply_angular_impulse(parent, axis, -impulse);
            } else if (angle > joint->upper_limit - PF_JOINT_LIMIT_SLOP && spin > 0.0f) {
                float impulse = -spin / mass;
                pf_joint_apply_angular_impulse(child, axis, impulse);
                if (parent != NULL) pf_joint_apply_angular_impulse(parent, axis, -impulse);
            }
        }
    }
}

__device__ static inline void pf_joint_solve_velocity(
        PfJoint* joints, int joint_count, PfBody* bodies) {
    for (int iteration = 0; iteration < PF_JOINT_VELOCITY_ITERATIONS; ++iteration)
        pf_joint_solve_velocity_once(joints, joint_count, bodies);
}

__device__ static inline void pf_joint_project_positions_once(
        PfJoint* joints, int joint_count, PfBody* bodies) {
    for (int index = 0; index < joint_count; ++index) {
        PfJoint* joint = &joints[index];
        if (!pf_joint_index_ordered(joint)) continue;
        PfBody* parent = joint->parent_body < 0 ? NULL
            : &bodies[joint->parent_body];
        PfBody* child = &bodies[joint->child_body];
        PfVec3 parent_anchor = parent == NULL ? joint->parent_anchor
            : pf_add(parent->position, pf_quat_rotate(parent->rotation,
                joint->parent_anchor));
        PfVec3 child_anchor = pf_add(child->position,
            pf_quat_rotate(child->rotation, joint->child_anchor));
        PfVec3 error = pf_sub(parent_anchor, child_anchor);
        PfVec3 directions[3] = {pf_v3(1.0f, 0.0f, 0.0f),
            pf_v3(0.0f, 1.0f, 0.0f), pf_v3(0.0f, 0.0f, 1.0f)};
        for (int row = 0; row < 3; ++row) {
            float distance = pf_dot(error, directions[row]);
            if (fabsf(distance) <= PF_JOINT_POSITION_SLOP) continue;
            float mass = 0.0f;
            if (parent != NULL && pf_mode_is_dynamic(parent->mode)) mass += parent->inverse_mass;
            if (pf_mode_is_dynamic(child->mode)) mass += child->inverse_mass;
            if (mass <= 1.0e-8f) continue;
            float correction = fmaxf(-PF_JOINT_MAX_POSITION_CORRECTION,
                fminf(PF_JOINT_MAX_POSITION_CORRECTION,
                    PF_JOINT_POSITION_PERCENT * distance));
            if (parent != NULL) pf_apply_position_correction(parent,
                parent_anchor, pf_scale(directions[row], -1.0f), correction
                * (pf_mode_is_dynamic(parent->mode)
                    ? parent->inverse_mass / mass : 0.0f));
            pf_apply_position_correction(child, child_anchor,
                directions[row], correction
                * (pf_mode_is_dynamic(child->mode)
                    ? child->inverse_mass / mass : 0.0f));
        }
        float angle = pf_joint_angle(*joint, bodies);
        float correction = angle < joint->lower_limit
            ? joint->lower_limit - angle
            : angle > joint->upper_limit ? joint->upper_limit - angle : 0.0f;
        if (correction != 0.0f) {
            PfVec3 axis = pf_joint_world_axis(*joint, bodies);
            float mass = 0.0f;
            if (parent != NULL && pf_mode_is_dynamic(parent->mode)) mass +=
                pf_dot(axis, pf_inverse_inertia_world(parent, axis));
            if (pf_mode_is_dynamic(child->mode)) mass +=
                pf_dot(axis, pf_inverse_inertia_world(child, axis));
            if (mass <= 1.0e-8f) continue;
            if (parent != NULL && pf_mode_is_dynamic(parent->mode)) {
                float share = -correction
                    * (pf_dot(axis, pf_inverse_inertia_world(parent, axis)) / mass);
                PfQuat delta = pf_quat_from_axis_angle(axis, share);
                PfVec3 anchor = pf_add(parent->position,
                    pf_quat_rotate(parent->rotation, joint->parent_anchor));
                parent->rotation = pf_quat_normalize(pf_quat_multiply(delta,
                    parent->rotation));
                PfVec3 rotated_anchor = pf_add(parent->position,
                    pf_quat_rotate(parent->rotation, joint->parent_anchor));
                parent->position = pf_add(parent->position,
                    pf_sub(anchor, rotated_anchor));
            }
            if (pf_mode_is_dynamic(child->mode)) child->rotation =
                pf_quat_normalize(pf_quat_multiply(pf_quat_from_axis_angle(axis,
                    correction * (pf_dot(axis, pf_inverse_inertia_world(child,
                        axis)) / mass)), child->rotation));
        }
    }
}

__device__ static inline void pf_joint_solve_positions(PfJoint* joints,
        int joint_count, PfBody* bodies) {
    for (int iteration = 0; iteration < PF_JOINT_POSITION_ITERATIONS; ++iteration) {
        pf_joint_project_positions_once(joints, joint_count, bodies);
    }
}

__device__ static inline void pf_joint_refresh_state(PfJoint* joints,
        int joint_count, PfBody* bodies) {
    for (int index = 0; index < joint_count; ++index) {
        PfJoint* joint = &joints[index];
        if (!pf_joint_index_ordered(joint)) continue;
        PfBody* parent = joint->parent_body < 0 ? NULL
            : &bodies[joint->parent_body];
        PfBody* child = &bodies[joint->child_body];
        joint->angle = pf_joint_angle(*joint, bodies);
        PfVec3 axis = pf_joint_world_axis(*joint, bodies);
        joint->angular_velocity = pf_dot(pf_sub(child->angular_velocity,
            parent == NULL ? pf_v3(0.0f, 0.0f, 0.0f) : parent->angular_velocity), axis);
    }
}
