#pragma once

#include "forward_kinematics.cuh"
#include "integrate.cuh"

typedef struct PfReducedWorkspace {
    float* matrix;
    float* rhs;
    float* solution;
    int* incoming;
    PfVec3* axes;
    PfVec3* anchors;
    PfVec3* position;
    PfQuat* rotation;
    PfVec3* linear_velocity;
    PfVec3* angular_velocity;
    PfVec3* linear_bias;
    PfVec3* angular_bias;
} PfReducedWorkspace;


__device__ static inline bool pf_reduced_body_valid(const PfBody* body) {
    return body != NULL && pf_mode_is_dynamic(body->mode)
        && pf_number(body->inverse_mass) && body->inverse_mass > 0.0f
        && pf_vec_valid(body->inverse_inertia_local)
        && body->inverse_inertia_local.x > 0.0f
        && body->inverse_inertia_local.y > 0.0f
        && body->inverse_inertia_local.z > 0.0f
        && pf_vec_valid(body->position) && pf_quat_valid(body->rotation)
        && pf_vec_valid(body->linear_velocity)
        && pf_vec_valid(body->angular_velocity);
}

/* World-space inertia multiply. PfBody keeps a diagonal body-frame inertia, so
 * I_world v = R (I_local (R^T v)) with I_local the reciprocal of the stored
 * inverse inertia. The reduced mass matrix needs I, never I^-1. */
__host__ __device__ static inline PfVec3 pf_inertia_world(
        const PfBody* body, PfVec3 value) {
    PfVec3 local = pf_quat_rotate(pf_quat_conjugate(body->rotation), value);
    local = pf_v3(local.x / body->inverse_inertia_local.x,
        local.y / body->inverse_inertia_local.y,
        local.z / body->inverse_inertia_local.z);
    return pf_quat_rotate(body->rotation, local);
}

__device__ static inline bool pf_reduced_workspace_bind(PfJointWorld world,
        bool free_base, PfReducedWorkspace* out,
        float local_matrix[PF_REDUCED_MAX_DOF][PF_REDUCED_MAX_DOF],
        float local_rhs[PF_REDUCED_MAX_DOF],
        float local_solution[PF_REDUCED_MAX_DOF],
        int local_incoming[PF_REDUCED_MAX_DOF],
        PfVec3 local_axes[PF_REDUCED_MAX_DOF],
        PfVec3 local_anchors[PF_REDUCED_MAX_DOF],
        PfVec3 local_position[PF_REDUCED_MAX_DOF],
        PfQuat local_rotation[PF_REDUCED_MAX_DOF],
        PfVec3 local_linear_velocity[PF_REDUCED_MAX_DOF],
        PfVec3 local_angular_velocity[PF_REDUCED_MAX_DOF],
        PfVec3 local_linear_bias[PF_REDUCED_MAX_DOF],
        PfVec3 local_angular_bias[PF_REDUCED_MAX_DOF]) {
    if (out == NULL) return false;
    const int joint_count = world.joint_count;
    const int body_count = world.body_count;
    const int dof = joint_count + (free_base ? PF_REDUCED_BASE_DOF : 0);
    size_t required = (size_t)dof * dof + 2 * dof
        + 20 * (size_t)(joint_count + 1) + 6 * (size_t)joint_count;
    if ((size_t)world.scratch_capacity < required) {
        if (dof > PF_REDUCED_MAX_DOF || body_count > PF_REDUCED_MAX_DOF) {
            return false;
        }
        out->matrix = &local_matrix[0][0];
        out->rhs = local_rhs;
        out->solution = local_solution;
        out->incoming = local_incoming;
        out->axes = local_axes;
        out->anchors = local_anchors;
        out->position = local_position;
        out->rotation = local_rotation;
        out->linear_velocity = local_linear_velocity;
        out->angular_velocity = local_angular_velocity;
        out->linear_bias = local_linear_bias;
        out->angular_bias = local_angular_bias;
        return true;
    }

    float* cursor = world.scratch;
    out->matrix = cursor;
    cursor += (size_t)dof * dof;
    out->rhs = cursor;
    cursor += dof;
    out->solution = cursor;
    cursor += dof;
    out->incoming = (int*)cursor;
    cursor += body_count;
    out->axes = (PfVec3*)cursor;
    cursor += 3 * joint_count;
    out->anchors = (PfVec3*)cursor;
    cursor += 3 * joint_count;
    out->position = (PfVec3*)cursor;
    cursor += 3 * body_count;
    out->rotation = (PfQuat*)cursor;
    cursor += 4 * body_count;
    out->linear_velocity = (PfVec3*)cursor;
    cursor += 3 * body_count;
    out->angular_velocity = (PfVec3*)cursor;
    cursor += 3 * body_count;
    out->linear_bias = (PfVec3*)cursor;
    cursor += 3 * body_count;
    out->angular_bias = (PfVec3*)cursor;
    return true;
}

__device__ static inline void pf_reduced_update_vectors(PfJointWorld world,
        PfReducedWorkspace workspace, bool free_base, PfVec3 position[],
        PfQuat rotation[]) {
    for (int joint = 0; joint < world.joint_count; ++joint) {
        const PfJoint* item = &world.joints[joint];
        if (item->parent_body < 0) {
            workspace.anchors[joint] = item->parent_anchor;
            workspace.axes[joint] = item->axis;
        } else {
            workspace.anchors[joint] = pf_add(position[item->parent_body],
                pf_quat_rotate(rotation[item->parent_body], item->parent_anchor));
            workspace.axes[joint] = pf_quat_rotate(
                rotation[item->parent_body], item->axis);
        }
    }
    (void)free_base;
}

__device__ static inline void pf_reduced_compute_velocities(
        PfJointWorld world, PfReducedWorkspace workspace, bool free_base) {
    for (int body = 0; body < world.body_count; ++body) {
        int joint = workspace.incoming[body];
        if (joint < 0) {
            workspace.linear_velocity[body] = free_base
                ? world.bodies[body].linear_velocity : pf_v3(0.0f, 0.0f, 0.0f);
            workspace.angular_velocity[body] = free_base
                ? world.bodies[body].angular_velocity : pf_v3(0.0f, 0.0f, 0.0f);
            continue;
        }
        const PfJoint* item = &world.joints[joint];
        PfVec3 relative_velocity = pf_scale(pf_cross(workspace.axes[joint],
            pf_sub(workspace.position[body], workspace.anchors[joint])),
            item->angular_velocity);
        if (item->parent_body < 0) {
            workspace.linear_velocity[body] = relative_velocity;
            workspace.angular_velocity[body] = pf_scale(workspace.axes[joint],
                item->angular_velocity);
        } else {
            int parent = item->parent_body;
            PfVec3 parent_offset = pf_sub(workspace.position[body],
                workspace.position[parent]);
            workspace.linear_velocity[body] = pf_add(
                workspace.linear_velocity[parent], pf_add(pf_cross(
                    workspace.angular_velocity[parent], parent_offset),
                    relative_velocity));
            workspace.angular_velocity[body] = pf_add(
                workspace.angular_velocity[parent],
                pf_scale(workspace.axes[joint], item->angular_velocity));
        }
    }
}

__device__ static inline void pf_reduced_compute_bias(PfJointWorld world,
        PfReducedWorkspace workspace) {
    /* qddot-free accelerations. With d_p the parent-COM to anchor lever,
     * r the child-COM to anchor lever, s the child-COM to parent-COM lever,
     * and a_dot = omega_parent x axis, differentiating
     *   v_i = v_p + omega_p x s + qdot (a x r)
     * gives
     *   alpha_i = alpha_p + qdot (omega_p x a)
     *   a_i     = a_p + alpha_p x s
     *             + omega_p x (omega_p x d_p)
     *             + qdot ((omega_p x a) x r) + omega_i x (omega_i x r)
     * and omega_i x r is exactly r_dot. The parent-transport and Coriolis
     * pair that the a x r form hides are already inside omega_i x (omega_i x r);
     * adding them again double counts. */
    for (int body = 0; body < world.body_count; ++body) {
        int joint = workspace.incoming[body];
        if (joint < 0) {
            workspace.linear_bias[body] = pf_v3(0.0f, 0.0f, 0.0f);
            workspace.angular_bias[body] = pf_v3(0.0f, 0.0f, 0.0f);
            continue;
        }
        const PfJoint* item = &world.joints[joint];
        PfVec3 axis = workspace.axes[joint];
        float speed = item->angular_velocity;
        PfVec3 lever = pf_sub(workspace.position[body], workspace.anchors[joint]);
        workspace.angular_bias[body] = pf_v3(0.0f, 0.0f, 0.0f);
        workspace.linear_bias[body] = pf_scale(pf_cross(axis,
            pf_cross(axis, lever)), speed * speed);
        if (item->parent_body < 0) continue;
        int parent = item->parent_body;
        PfVec3 parent_lever = pf_sub(workspace.position[body],
            workspace.position[parent]);
        PfVec3 anchor_lever = pf_sub(workspace.anchors[joint],
            workspace.position[parent]);
        PfVec3 parent_spin = workspace.angular_velocity[parent];
        PfVec3 child_spin = workspace.angular_velocity[body];
        PfVec3 axis_rate = pf_cross(parent_spin, axis);
        workspace.angular_bias[body] = pf_add(workspace.angular_bias[parent],
            pf_scale(axis_rate, speed));
        workspace.linear_bias[body] = pf_add(workspace.linear_bias[parent],
            pf_add(pf_cross(workspace.angular_bias[parent], parent_lever),
            pf_add(pf_cross(parent_spin,
                pf_cross(parent_spin, anchor_lever)),
            pf_add(pf_scale(pf_cross(axis_rate, lever), speed),
                pf_cross(child_spin,
                    pf_cross(child_spin, lever))))));
    }
}

__device__ static inline bool pf_reduced_prepare(PfJointWorld world,
        bool free_base, PfReducedWorkspace* workspace,
        float local_matrix[PF_REDUCED_MAX_DOF][PF_REDUCED_MAX_DOF],
        float local_rhs[PF_REDUCED_MAX_DOF],
        float local_solution[PF_REDUCED_MAX_DOF],
        int local_incoming[PF_REDUCED_MAX_DOF],
        PfVec3 local_axes[PF_REDUCED_MAX_DOF],
        PfVec3 local_anchors[PF_REDUCED_MAX_DOF],
        PfVec3 local_position[PF_REDUCED_MAX_DOF],
        PfQuat local_rotation[PF_REDUCED_MAX_DOF],
        PfVec3 local_linear_velocity[PF_REDUCED_MAX_DOF],
        PfVec3 local_angular_velocity[PF_REDUCED_MAX_DOF],
        PfVec3 local_linear_bias[PF_REDUCED_MAX_DOF],
        PfVec3 local_angular_bias[PF_REDUCED_MAX_DOF]) {
    if (world.joints == NULL || world.bodies == NULL
            || world.joint_count <= 0 || world.joint_capacity <= 0
            || world.body_count <= 0 || world.body_capacity <= 0
            || world.scratch_capacity < 0
            || (world.scratch_capacity > 0 && world.scratch == NULL)
            || world.joint_count > world.joint_capacity
            || world.body_count < world.joint_count
            || world.body_count > world.body_capacity
            || world.body_count != world.joint_count + (free_base ? 1 : 0)) {
        return false;
    }
    for (int body = 0; body < world.body_count; ++body) {
        if (!pf_reduced_body_valid(&world.bodies[body])) return false;
    }
    for (int joint = 0; joint < world.joint_count; ++joint) {
        const PfJoint* item = &world.joints[joint];
        if (!pf_joint_valid(item, world.body_count)
                || (free_base && (item->parent_body < 0 || item->child_body == 0))) {
            return false;
        }
    }
    if (!pf_reduced_workspace_bind(world, free_base, workspace, local_matrix,
            local_rhs, local_solution, local_incoming, local_axes,
            local_anchors, local_position, local_rotation,
            local_linear_velocity, local_angular_velocity, local_linear_bias,
            local_angular_bias)) {
        return false;
    }

    for (int body = 0; body < world.body_count; ++body) {
        workspace->incoming[body] = -1;
        workspace->position[body] = world.bodies[body].position;
        workspace->rotation[body] = world.bodies[body].rotation;
    }
    for (int joint = 0; joint < world.joint_count; ++joint) {
        int child = world.joints[joint].child_body;
        if (workspace->incoming[child] >= 0) return false;
        workspace->incoming[child] = joint;
    }
    for (int body = 0; body < world.body_count; ++body) {
        if ((free_base && body == 0 && workspace->incoming[body] >= 0)
                || (!free_base && workspace->incoming[body] < 0)
                || (free_base && body > 0 && workspace->incoming[body] < 0)) {
            return false;
        }
    }
    for (int joint = 0; joint < world.joint_count; ++joint) {
        int parent = world.joints[joint].parent_body;
        if (parent >= 0 && workspace->incoming[parent] >= joint) return false;
    }

    for (int joint = 0; joint < world.joint_count; ++joint) {
        const PfJoint* item = &world.joints[joint];
        int child = item->child_body;
        PfQuat delta = pf_quat_from_axis_angle(item->axis, item->angle);
        if (item->parent_body < 0) {
            workspace->rotation[child] = delta;
            workspace->position[child] = pf_sub(item->parent_anchor,
                pf_quat_rotate(delta, item->child_anchor));
        } else {
            int parent = item->parent_body;
            workspace->rotation[child] = pf_quat_normalize(
                pf_quat_multiply(workspace->rotation[parent], delta));
            PfVec3 anchor = pf_add(workspace->position[parent],
                pf_quat_rotate(workspace->rotation[parent], item->parent_anchor));
            workspace->position[child] = pf_sub(anchor,
                pf_quat_rotate(workspace->rotation[child], item->child_anchor));
        }
        if (!pf_vec_valid(workspace->position[child])
                || !pf_quat_valid(workspace->rotation[child])) {
            return false;
        }
    }
    pf_reduced_update_vectors(world, *workspace, free_base,
        workspace->position, workspace->rotation);
    pf_reduced_compute_velocities(world, *workspace, free_base);
    return true;
}

__device__ static inline bool pf_reduced_joint_upstream(
        const PfJointWorld world, const PfReducedWorkspace workspace,
        int body, int target) {
    while (body >= 0) {
        int joint = workspace.incoming[body];
        if (joint < 0) return false;
        if (joint == target) return true;
        body = world.joints[joint].parent_body;
    }
    return false;
}

__device__ static inline PfVec3 pf_reduced_linear_column(PfJointWorld world,
        PfReducedWorkspace workspace, bool free_base, int row, int body) {
    if (free_base) {
        if (row >= PF_REDUCED_BASE_DOF) {
            int joint = row - PF_REDUCED_BASE_DOF;
            return pf_reduced_joint_upstream(world, workspace, body, joint)
                ? pf_cross(workspace.axes[joint], pf_sub(workspace.position[body],
                    workspace.anchors[joint])) : pf_v3(0.0f, 0.0f, 0.0f);
        }
        if (row >= 3) {
            /* Base spin transports every upstream body about the base centre
             * of mass, so its linear column is axis x (x_body - x_base). */
            PfVec3 axis = pf_v3(row == 3 ? 1.0f : 0.0f,
                row == 4 ? 1.0f : 0.0f, row == 5 ? 1.0f : 0.0f);
            return pf_cross(axis, pf_sub(workspace.position[body],
                workspace.position[0]));
        }
        return pf_v3(row == 0 ? 1.0f : 0.0f, row == 1 ? 1.0f : 0.0f,
            row == 2 ? 1.0f : 0.0f);
    }
    return pf_reduced_joint_upstream(world, workspace, body, row)
        ? pf_cross(workspace.axes[row], pf_sub(workspace.position[body],
            workspace.anchors[row])) : pf_v3(0.0f, 0.0f, 0.0f);
}

__device__ static inline PfVec3 pf_reduced_angular_column(PfJointWorld world,
        PfReducedWorkspace workspace, bool free_base, int row, int body) {
    if (free_base) {
        if (row < 3) return pf_v3(0.0f, 0.0f, 0.0f);
        if (row < PF_REDUCED_BASE_DOF) {
            int component = row - 3;
            return pf_v3(component == 0 ? 1.0f : 0.0f,
                component == 1 ? 1.0f : 0.0f,
                component == 2 ? 1.0f : 0.0f);
        }
        int joint = row - PF_REDUCED_BASE_DOF;
        return pf_reduced_joint_upstream(world, workspace, body, joint)
            ? workspace.axes[joint] : pf_v3(0.0f, 0.0f, 0.0f);
    }
    return pf_reduced_joint_upstream(world, workspace, body, row)
        ? workspace.axes[row] : pf_v3(0.0f, 0.0f, 0.0f);
}

__device__ static inline void pf_reduced_assemble(PfJointWorld world,
        bool free_base, PfReducedWorkspace workspace, PfVec3 gravity,
        bool bias, float rhs[]) {
    const int dof = world.joint_count + (free_base ? PF_REDUCED_BASE_DOF : 0);
    for (int index = 0; index < dof * dof; ++index) workspace.matrix[index] = 0.0f;
    for (int index = 0; index < dof; ++index) rhs[index] = 0.0f;

    for (int body_index = 0; body_index < world.body_count; ++body_index) {
        const PfBody* source = &world.bodies[body_index];
        PfBody body = *source;
        body.rotation = workspace.rotation[body_index];
        float mass = 1.0f / body.inverse_mass;
        PfVec3 spin = workspace.angular_velocity[body_index];
        /* Newton-Euler residual torque, a bias term not a load. */
        PfVec3 gyro = bias
            ? pf_cross(spin, pf_inertia_world(&body, spin)) : pf_v3(0, 0, 0);
        for (int row = 0; row < dof; ++row) {
            PfVec3 row_linear = pf_reduced_linear_column(world, workspace,
                free_base, row, body_index);
            PfVec3 row_angular = pf_reduced_angular_column(world, workspace,
                free_base, row, body_index);
            PfVec3 weighted_linear = pf_scale(row_linear, mass);
            PfVec3 weighted_angular = pf_inertia_world(&body, row_angular);
            for (int column = 0; column <= row; ++column) {
                PfVec3 column_linear = pf_reduced_linear_column(world, workspace,
                    free_base, column, body_index);
                PfVec3 column_angular = pf_reduced_angular_column(world,
                    workspace, free_base, column, body_index);
                float value = pf_dot(weighted_linear, column_linear)
                    + pf_dot(weighted_angular, column_angular);
                workspace.matrix[row * dof + column] += value;
                if (row != column) {
                    workspace.matrix[column * dof + row] += value;
                }
            }
            /* Gravity enters once; body force and torque are projected like
             * any other applied load and belong to the caller's substep. */
            rhs[row] += pf_dot(weighted_linear, gravity)
                + pf_dot(row_linear, source->force)
                + pf_dot(row_angular, source->torque);
            if (bias) {
                /* The angular bias is an ACCELERATION, so it reaches the
                 * generalised force through the same inertia the mass matrix
                 * uses: Jw^T (I alpha_bias + omega x I omega). Scaling only
                 * the linear side by mass while leaving alpha_bias unscaled
                 * is dimensionally wrong by a factor of I. */
                rhs[row] -= pf_dot(weighted_linear,
                    workspace.linear_bias[body_index])
                    + pf_dot(weighted_angular,
                        workspace.angular_bias[body_index])
                    + pf_dot(row_angular, gyro);
            }
        }
    }
    for (int joint = 0; joint < world.joint_count; ++joint) {
        const PfJoint* item = &world.joints[joint];
        int row = (free_base ? PF_REDUCED_BASE_DOF : 0) + joint;
        workspace.matrix[row * dof + row] += item->armature;
        float motor = item->motor_stiffness * (item->motor_target
            - item->angle) - item->motor_damping * item->angular_velocity;
        rhs[row] += fmaxf(-item->motor_max_torque,
            fminf(item->motor_max_torque, motor)) - item->damping
            * item->angular_velocity;
    }
}

__device__ static inline bool pf_reduced_solve(PfReducedWorkspace workspace,
        int dof) {
    for (int row = 0; row < dof; ++row) {
        for (int column = 0; column <= row; ++column) {
            float value = workspace.matrix[row * dof + column];
            for (int inner = 0; inner < column; ++inner) {
                value -= workspace.matrix[row * dof + inner]
                    * workspace.matrix[column * dof + inner];
            }
            if (!pf_number(value)) return false;
            if (row == column) {
                if (value <= 1.0e-12f) return false;
                workspace.matrix[row * dof + column] = sqrtf(value);
            } else {
                workspace.matrix[row * dof + column] = value
                    / workspace.matrix[column * dof + column];
            }
        }
    }
    for (int row = 0; row < dof; ++row) {
        float value = workspace.rhs[row];
        if (!pf_number(value)) return false;
        for (int column = 0; column < row; ++column) {
            value -= workspace.matrix[row * dof + column]
                * workspace.solution[column];
        }
        workspace.solution[row] = value / workspace.matrix[row * dof + row];
    }
    for (int row = dof - 1; row >= 0; --row) {
        float value = workspace.solution[row];
        for (int column = row + 1; column < dof; ++column) {
            value -= workspace.matrix[column * dof + row]
                * workspace.solution[column];
        }
        workspace.solution[row] = value / workspace.matrix[row * dof + row];
        if (!pf_number(workspace.solution[row])) return false;
    }
    return true;
}

__device__ static inline bool pf_reduced_dynamics_step(PfJointWorld world,
        PfVec3 gravity, float dt) {
    if (!pf_vec_valid(gravity) || !pf_number(dt) || dt <= 0.0f
            || world.joint_count <= 0 || world.body_count <= 0
            || (world.body_count != world.joint_count
                && world.body_count != world.joint_count + 1)) {
        return false;
    }
    const bool free_base = world.body_count == world.joint_count + 1;
    float local_matrix[PF_REDUCED_MAX_DOF][PF_REDUCED_MAX_DOF];
    float local_rhs[PF_REDUCED_MAX_DOF];
    float local_solution[PF_REDUCED_MAX_DOF];
    int local_incoming[PF_REDUCED_MAX_DOF];
    PfVec3 local_axes[PF_REDUCED_MAX_DOF];
    PfVec3 local_anchors[PF_REDUCED_MAX_DOF];
    PfVec3 local_position[PF_REDUCED_MAX_DOF];
    PfQuat local_rotation[PF_REDUCED_MAX_DOF];
    PfVec3 local_linear_velocity[PF_REDUCED_MAX_DOF];
    PfVec3 local_angular_velocity[PF_REDUCED_MAX_DOF];
    PfVec3 local_linear_bias[PF_REDUCED_MAX_DOF];
    PfVec3 local_angular_bias[PF_REDUCED_MAX_DOF];
    PfReducedWorkspace workspace = {};
    if (!pf_reduced_prepare(world, free_base, &workspace, local_matrix,
            local_rhs, local_solution, local_incoming, local_axes,
            local_anchors, local_position, local_rotation,
            local_linear_velocity, local_angular_velocity, local_linear_bias,
            local_angular_bias)) {
        return false;
    }
    pf_reduced_compute_bias(world, workspace);
    const int dof = world.joint_count + (free_base ? PF_REDUCED_BASE_DOF : 0);
    pf_reduced_assemble(world, free_base, workspace, gravity, true,
        workspace.rhs);
    if (!pf_reduced_solve(workspace, dof)) {
        return false;
    }

    PfVec3 base_linear = world.bodies[0].linear_velocity;
    PfVec3 base_angular = world.bodies[0].angular_velocity;
    if (free_base) {
        base_linear = pf_add(base_linear, pf_scale(pf_v3(workspace.solution[0],
            workspace.solution[1], workspace.solution[2]), dt));
        base_angular = pf_add(base_angular, pf_scale(pf_v3(workspace.solution[3],
            workspace.solution[4], workspace.solution[5]), dt));
        if (!pf_vec_valid(base_linear) || !pf_vec_valid(base_angular)
                || !pf_vec_valid(pf_add(world.bodies[0].position,
                    pf_scale(base_linear, dt)))) {
            return false;
        }
    }
    for (int joint = 0; joint < world.joint_count; ++joint) {
        const PfJoint* item = &world.joints[joint];
        float velocity = item->angular_velocity
            + workspace.solution[(free_base ? PF_REDUCED_BASE_DOF : 0) + joint] * dt;
        float angle = item->angle + velocity * dt;
        if (!pf_number(velocity) || !pf_number(angle)) return false;
    }

    if (free_base) {
        PfBody* base = &world.bodies[0];
        base->linear_velocity = base_linear;
        base->angular_velocity = base_angular;
        base->position = pf_add(base->position, pf_scale(base_linear, dt));
        base->rotation = pf_quat_normalize(pf_quat_multiply(
            pf_orientation_delta(base_angular, dt), base->rotation));
    }
    for (int joint = 0; joint < world.joint_count; ++joint) {
        PfJoint* item = &world.joints[joint];
        item->angular_velocity += workspace.solution[
            (free_base ? PF_REDUCED_BASE_DOF : 0) + joint] * dt;
        item->angle += item->angular_velocity * dt;
        if (item->angle < item->lower_limit) {
            item->angle = item->lower_limit;
            if (item->angular_velocity < 0.0f) item->angular_velocity = 0.0f;
        } else if (item->angle > item->upper_limit) {
            item->angle = item->upper_limit;
            if (item->angular_velocity > 0.0f) item->angular_velocity = 0.0f;
        }
    }
    (void)pf_joint_forward_kinematics(world);
    for (int body = 0; body < world.body_count; ++body) {
        workspace.position[body] = world.bodies[body].position;
        workspace.rotation[body] = world.bodies[body].rotation;
    }
    pf_reduced_update_vectors(world, workspace, free_base, workspace.position,
        workspace.rotation);
    pf_reduced_compute_velocities(world, workspace, free_base);
    for (int body = 0; body < world.body_count; ++body) {
        world.bodies[body].linear_velocity = workspace.linear_velocity[body];
        world.bodies[body].angular_velocity = workspace.angular_velocity[body];
    }
    return true;
}

__device__ static inline bool pf_reduced_mass_matrix(PfJointWorld world,
        float* matrix, int matrix_capacity) {
    if (matrix == NULL || world.joint_count <= 0) return false;
    const bool free_base = world.body_count == world.joint_count + 1;
    const int dof = world.joint_count + (free_base ? PF_REDUCED_BASE_DOF : 0);
    if (dof <= 0 || matrix_capacity < dof * dof) return false;
    float local_matrix[PF_REDUCED_MAX_DOF][PF_REDUCED_MAX_DOF];
    float local_rhs[PF_REDUCED_MAX_DOF];
    float local_solution[PF_REDUCED_MAX_DOF];
    int local_incoming[PF_REDUCED_MAX_DOF];
    PfVec3 local_axes[PF_REDUCED_MAX_DOF];
    PfVec3 local_anchors[PF_REDUCED_MAX_DOF];
    PfVec3 local_position[PF_REDUCED_MAX_DOF];
    PfQuat local_rotation[PF_REDUCED_MAX_DOF];
    PfVec3 local_linear_velocity[PF_REDUCED_MAX_DOF];
    PfVec3 local_angular_velocity[PF_REDUCED_MAX_DOF];
    PfVec3 local_linear_bias[PF_REDUCED_MAX_DOF];
    PfVec3 local_angular_bias[PF_REDUCED_MAX_DOF];
    PfReducedWorkspace workspace = {};
    if (!pf_reduced_prepare(world, free_base, &workspace, local_matrix,
            local_rhs, local_solution, local_incoming, local_axes,
            local_anchors, local_position, local_rotation,
            local_linear_velocity, local_angular_velocity, local_linear_bias,
            local_angular_bias)) {
        return false;
    }
    pf_reduced_assemble(world, free_base, workspace,
        pf_v3(0.0f, 0.0f, 0.0f), false, workspace.rhs);
    for (int index = 0; index < dof * dof; ++index) matrix[index] = workspace.matrix[index];
    return true;
}

__device__ static inline bool pf_reduced_planar_chain_mass(PfJointWorld world,
        float matrix[PF_REDUCED_MAX_BODIES][PF_REDUCED_MAX_BODIES]) {
    if (matrix == NULL || world.body_count != world.joint_count
            || world.joint_count <= 0 || world.joint_count > PF_REDUCED_MAX_BODIES) {
        return false;
    }
    float local_matrix[PF_REDUCED_MAX_DOF][PF_REDUCED_MAX_DOF];
    float local_rhs[PF_REDUCED_MAX_DOF];
    float local_solution[PF_REDUCED_MAX_DOF];
    int local_incoming[PF_REDUCED_MAX_DOF];
    PfVec3 local_axes[PF_REDUCED_MAX_DOF];
    PfVec3 local_anchors[PF_REDUCED_MAX_DOF];
    PfVec3 local_position[PF_REDUCED_MAX_DOF];
    PfQuat local_rotation[PF_REDUCED_MAX_DOF];
    PfVec3 local_linear_velocity[PF_REDUCED_MAX_DOF];
    PfVec3 local_angular_velocity[PF_REDUCED_MAX_DOF];
    PfVec3 local_linear_bias[PF_REDUCED_MAX_DOF];
    PfVec3 local_angular_bias[PF_REDUCED_MAX_DOF];
    PfReducedWorkspace workspace = {};
    if (!pf_reduced_prepare(world, false, &workspace, local_matrix,
            local_rhs, local_solution, local_incoming, local_axes,
            local_anchors, local_position, local_rotation,
            local_linear_velocity, local_angular_velocity, local_linear_bias,
            local_angular_bias)) {
        return false;
    }
    pf_reduced_assemble(world, false, workspace,
        pf_v3(0.0f, 0.0f, 0.0f), false, workspace.rhs);
    for (int row = 0; row < PF_REDUCED_MAX_BODIES; ++row) {
        for (int column = 0; column < PF_REDUCED_MAX_BODIES; ++column) {
            matrix[row][column] = row < world.joint_count
                && column < world.joint_count
                ? workspace.matrix[row * world.joint_count + column] : 0.0f;
        }
    }
    (void)pf_joint_forward_kinematics(world);
    return true;
}


__device__ static inline bool pf_reduced_planar_chain_step(
        PfJointWorld world, float gravity, float dt) {
    if (!pf_number(gravity)) return false;
    return pf_reduced_dynamics_step(world, pf_v3(0.0f, gravity, 0.0f), dt);
}

__device__ static inline bool pf_reduced_forward_dynamics_step(
        PfJointWorld world, PfVec3 gravity, float dt) {
    return pf_reduced_dynamics_step(world, gravity, dt);
}

__global__ static void pf_reduced_forward_dynamics_kernel(
        PfJoint* joints, PfBody* bodies, int env_count, int joint_stride,
        int body_stride, PfVec3 gravity, float dt, float* output) {
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) return;
    PfJointWorld world = {};
    world.joints = joints + (size_t)env * joint_stride;
    world.bodies = bodies + (size_t)env * body_stride;
    world.joint_count = joint_stride;
    world.body_count = body_stride;
    world.joint_capacity = joint_stride;
    world.body_capacity = body_stride;
    bool ok = pf_reduced_forward_dynamics_step(world, gravity, dt);
    if (output != NULL) output[env] = ok ? world.bodies[0].position.y : 0.0f;
}
