#pragma once

#include "math.cuh"

__device__ static inline PfQuat pf_orientation_delta(
        PfVec3 angular_velocity, float dt) {
    float half_angle = 0.5f * pf_length(angular_velocity) * dt;
    float scale = half_angle > 1.0e-8f ? sinf(half_angle)
        / pf_length(angular_velocity) : 0.5f * dt;
    PfQuat result = {cosf(half_angle), angular_velocity.x * scale,
        angular_velocity.y * scale, angular_velocity.z * scale};
    return result;
}

__device__ static inline void pf_integrate(
        PfBody* bodies, int body_count, PfVec3 gravity, float dt) {
    for (int index = 0; index < body_count; ++index) {
        PfBody* body = &bodies[index];
        if (body->mode == PF_DYNAMIC) {
            body->linear_velocity.x += (gravity.x
                + body->force.x * body->inverse_mass) * dt;
            body->linear_velocity.y += (gravity.y
                + body->force.y * body->inverse_mass) * dt;
            body->linear_velocity.z += (gravity.z
                + body->force.z * body->inverse_mass) * dt;
            if (body->torque.x != 0.0f || body->torque.y != 0.0f
                    || body->torque.z != 0.0f) {
                body->angular_velocity = pf_add(body->angular_velocity,
                    pf_scale(pf_inverse_inertia_world(body, body->torque), dt));
            }
        } else if (body->mode != PF_KINEMATIC) {
            continue;
        }
        body->position.x += body->linear_velocity.x * dt;
        body->position.y += body->linear_velocity.y * dt;
        body->position.z += body->linear_velocity.z * dt;
        if (body->angular_velocity.x != 0.0f
                || body->angular_velocity.y != 0.0f
                || body->angular_velocity.z != 0.0f) {
            PfQuat delta = pf_orientation_delta(body->angular_velocity, dt);
            body->rotation = pf_quat_normalize(pf_quat_multiply(delta,
                body->rotation));
        } else {
            float norm = body->rotation.w * body->rotation.w
                + body->rotation.x * body->rotation.x
                + body->rotation.y * body->rotation.y
                + body->rotation.z * body->rotation.z;
            if (norm < 0.999999f || norm > 1.000001f) {
                body->rotation = pf_quat_normalize(body->rotation);
            }
        }
    }
}

__device__ static inline void pf_clear_forces(PfBody* bodies, int body_count) {
    for (int index = 0; index < body_count; ++index) {
        bodies[index].force = pf_v3(0.0f, 0.0f, 0.0f);
        bodies[index].torque = pf_v3(0.0f, 0.0f, 0.0f);
    }
}
