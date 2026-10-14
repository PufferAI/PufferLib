#pragma once

#include "types.cuh"

__host__ __device__ static inline PfVec3 pf_v3(float x, float y, float z) {
    PfVec3 result = {x, y, z};
    return result;
}

__host__ __device__ static inline PfVec3 pf_add(PfVec3 a, PfVec3 b) {
    return pf_v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ static inline PfVec3 pf_sub(PfVec3 a, PfVec3 b) {
    return pf_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ static inline PfVec3 pf_scale(PfVec3 value, float scale) {
    return pf_v3(value.x * scale, value.y * scale, value.z * scale);
}

__host__ __device__ static inline float pf_dot(PfVec3 a, PfVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ static inline PfVec3 pf_cross(PfVec3 a, PfVec3 b) {
    return pf_v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__host__ __device__ static inline float pf_length_squared(PfVec3 value) {
    return pf_dot(value, value);
}

__host__ __device__ static inline float pf_length(PfVec3 value) {
    return sqrtf(pf_length_squared(value));
}

__host__ __device__ static inline PfVec3 pf_normalize_or(
        PfVec3 value, PfVec3 fallback) {
    float length_squared = pf_length_squared(value);
    if (length_squared > 1.0e-20f) {
        return pf_scale(value, rsqrtf(length_squared));
    }
    length_squared = pf_length_squared(fallback);
    return length_squared > 1.0e-20f
        ? pf_scale(fallback, rsqrtf(length_squared)) : pf_v3(1.0f, 0.0f, 0.0f);
}

__host__ __device__ static inline PfQuat pf_quat_conjugate(PfQuat value) {
    PfQuat result = {value.w, -value.x, -value.y, -value.z};
    return result;
}

__host__ __device__ static inline PfQuat pf_quat_multiply(PfQuat a, PfQuat b) {
    PfQuat result;
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return result;
}

__host__ __device__ static inline PfVec3 pf_quat_rotate(PfQuat q, PfVec3 value) {
    PfVec3 qv = pf_v3(q.x, q.y, q.z);
    PfVec3 t = pf_scale(pf_cross(qv, value), 2.0f);
    return pf_add(pf_add(value, pf_scale(t, q.w)), pf_cross(qv, t));
}

__host__ __device__ static inline PfQuat pf_quat_normalize(PfQuat q) {
    float length_squared = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (length_squared <= 1.0e-20f) {
        return pf_quat_identity();
    }
    float inverse = rsqrtf(length_squared);
    PfQuat result = {q.w * inverse, q.x * inverse, q.y * inverse, q.z * inverse};
    return result;
}

__host__ __device__ static inline PfQuat pf_quat_from_axis_angle(
        PfVec3 axis, float angle) {
    PfVec3 unit = pf_normalize_or(axis, pf_v3(0.0f, 1.0f, 0.0f));
    float half = 0.5f * angle;
    float sine = sinf(half);
    PfQuat result = {cosf(half), unit.x * sine, unit.y * sine, unit.z * sine};
    return result;
}

__host__ __device__ static inline void pf_quat_axes(
        PfQuat rotation, PfVec3 axes[3]) {
    axes[0] = pf_quat_rotate(rotation, pf_v3(1.0f, 0.0f, 0.0f));
    axes[1] = pf_quat_rotate(rotation, pf_v3(0.0f, 1.0f, 0.0f));
    axes[2] = pf_quat_rotate(rotation, pf_v3(0.0f, 0.0f, 1.0f));
}

__host__ __device__ static inline PfVec3 pf_inverse_inertia_world(
        const PfBody* body, PfVec3 value) {
    PfVec3 local = pf_quat_rotate(pf_quat_conjugate(body->rotation), value);
    local = pf_v3(
        local.x * body->inverse_inertia_local.x,
        local.y * body->inverse_inertia_local.y,
        local.z * body->inverse_inertia_local.z);
    return pf_quat_rotate(body->rotation, local);
}

__host__ __device__ static inline void pf_apply_impulse(
        PfBody* body, PfVec3 point, PfVec3 impulse) {
    if (!pf_mode_is_dynamic(body->mode)) {
        return;
    }
    body->linear_velocity = pf_add(body->linear_velocity,
        pf_scale(impulse, body->inverse_mass));
    body->angular_velocity = pf_add(body->angular_velocity,
        pf_inverse_inertia_world(body, pf_cross(pf_sub(point, body->position), impulse)));
}

__host__ __device__ static inline PfVec3 pf_point_velocity(
        const PfBody* body, PfVec3 point) {
    if (!pf_mode_is_movable(body->mode)) {
        return pf_v3(0.0f, 0.0f, 0.0f);
    }
    return pf_add(body->linear_velocity,
        pf_cross(body->angular_velocity, pf_sub(point, body->position)));
}

__host__ __device__ static inline float pf_radius(const PfBody* body) {
    if (body->shape == PF_SPHERE) {
        return body->half_extents.x;
    }
    if (body->shape == PF_CAPSULE) return body->half_extents.x + body->half_extents.y;
    if (body->shape == PF_CYLINDER) return sqrtf(body->half_extents.x * body->half_extents.x
        + body->half_extents.y * body->half_extents.y);
    return pf_length(body->half_extents);
}

__host__ __device__ static inline float pf_effective_mass(
        const PfBody* a, const PfBody* b, PfVec3 point_a, PfVec3 point_b,
        PfVec3 direction) {
    float result = 0.0f;
    if (pf_mode_is_dynamic(a->mode)) {
        PfVec3 lever = pf_cross(pf_sub(point_a, a->position), direction);
        result += a->inverse_mass + pf_dot(lever, pf_inverse_inertia_world(a, lever));
    }
    if (pf_mode_is_dynamic(b->mode)) {
        PfVec3 lever = pf_cross(pf_sub(point_b, b->position), direction);
        result += b->inverse_mass + pf_dot(lever, pf_inverse_inertia_world(b, lever));
    }
    return result;
}

__host__ __device__ static inline void pf_apply_position_correction(
        PfBody* body, PfVec3 point, PfVec3 direction, float magnitude) {
    (void)point;
    if (!pf_mode_is_dynamic(body->mode) || magnitude == 0.0f) {
        return;
    }
    PfVec3 displacement = pf_scale(direction, magnitude);
    body->position = pf_add(body->position, displacement);
}
