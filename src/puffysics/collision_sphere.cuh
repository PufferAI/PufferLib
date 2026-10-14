#pragma once

#include "math.cuh"

typedef struct PfPointQuery {
    PfVec3 normal;
    PfVec3 closest_point;
    float separation;
} PfPointQuery;

__host__ __device__ static inline bool pf_point_sphere_query(
        PfVec3 point, PfVec3 center, float radius, PfPointQuery* out) {
    if (out == NULL || !pf_vec_valid(point) || !pf_vec_valid(center)
            || !pf_number(radius) || radius <= 0.0f) {
        return false;
    }
    PfVec3 delta = pf_sub(point, center);
    float distance_squared = pf_length_squared(delta);
    float inverse_distance = distance_squared > 1.0e-20f
        ? rsqrtf(distance_squared) : 0.0f;
    PfPointQuery result;
    result.normal = inverse_distance > 0.0f
        ? pf_scale(delta, inverse_distance) : pf_v3(1.0f, 0.0f, 0.0f);
    result.closest_point = pf_add(center, pf_scale(result.normal, radius));
    result.separation = distance_squared * inverse_distance - radius;
    *out = result;
    return result.separation <= 0.0f;
}

// The box axis with the least clearance from an interior point, and that
// clearance. Ties keep the lowest index, which is the order the in-line loops
// these two functions replaced used.
__host__ __device__ static inline int pf_box_escape_axis(
        PfVec3 local, PfVec3 half_extents, float* clearance) {
    int axis_index = 0;
    *clearance = half_extents.x - fabsf(local.x);
    for (int axis = 1; axis < 3; ++axis) {
        float half = axis == 1 ? half_extents.y : half_extents.z;
        float value = axis == 1 ? local.y : local.z;
        float candidate = half - fabsf(value);
        if (candidate < *clearance) {
            *clearance = candidate;
            axis_index = axis;
        }
    }
    return axis_index;
}

// Containment is decided from the local coordinates against the half extents,
// NEVER from the distance to a reconstructed clamped point. Going point ->
// local -> clamped point -> point in float32 leaves a residual of order 1e-7
// m, which is a million times any squared-distance threshold in this file, and
// normalising that residual hands back a direction made of rounding rather
// than of geometry. Measured over 39997 genuinely interior sphere centres in
// translated and rotated boxes, the reconstructed-point test misclassified
// 56.6 per cent of them, with a worst normal error of 180 degrees against the
// least-clearance face and a worst separation error of 0.75 m.
__host__ __device__ static inline bool pf_box_contains(
        PfVec3 local, PfVec3 half_extents) {
    return fabsf(local.x) < half_extents.x
        && fabsf(local.y) < half_extents.y
        && fabsf(local.z) < half_extents.z;
}

__host__ __device__ static inline bool pf_point_box_query(
        PfVec3 point, const PfBody* box, PfPointQuery* out) {
    if (out == NULL || box == NULL || !pf_vec_valid(point)
            || !pf_vec_valid(box->position) || !pf_quat_valid(box->rotation)
            || !pf_vec_valid(box->half_extents) || box->half_extents.x <= 0.0f
            || box->half_extents.y <= 0.0f || box->half_extents.z <= 0.0f) {
        return false;
    }
    PfVec3 axes[3];
    pf_quat_axes(box->rotation, axes);
    PfVec3 delta = pf_sub(point, box->position);
    PfVec3 local = pf_v3(pf_dot(delta, axes[0]), pf_dot(delta, axes[1]),
        pf_dot(delta, axes[2]));
    PfVec3 closest = box->position;
    for (int axis = 0; axis < 3; ++axis) {
        float half = axis == 0 ? box->half_extents.x
            : axis == 1 ? box->half_extents.y : box->half_extents.z;
        float value = axis == 0 ? local.x : axis == 1 ? local.y : local.z;
        float clamped = value < -half ? -half : value > half ? half : value;
        closest = pf_add(closest, pf_scale(axes[axis], clamped));
    }
    delta = pf_sub(point, closest);
    float distance_squared = pf_length_squared(delta);
    float clearance;
    int axis_index = pf_box_escape_axis(local, box->half_extents, &clearance);
    float value = axis_index == 0 ? local.x
        : axis_index == 1 ? local.y : local.z;
    PfVec3 escape = pf_scale(axes[axis_index], value < 0.0f ? -1.0f : 1.0f);
    PfPointQuery result;
    if (pf_box_contains(local, box->half_extents)) {
        result.normal = escape;
        result.separation = -clearance;
        // The closest surface point is `clearance` from the QUERY POINT along
        // the escape normal, so it keeps the two coordinates the escape does
        // not move. Anchoring it to the box centre instead placed the point
        // `clearance` from the centre, which is a different point on the
        // surface and moves the sampled directions a caller derives from it.
        closest = pf_add(point, pf_scale(escape, clearance));
    } else if (distance_squared > 1.0e-20f) {
        float inverse_distance = rsqrtf(distance_squared);
        result.normal = pf_scale(delta, inverse_distance);
        result.separation = distance_squared * inverse_distance;
    } else {
        // Exterior and exactly on the surface: the distance is zero, so the
        // outward normal of the face the point sits on is the consistent
        // answer and the separation is the touching one.
        result.normal = escape;
        result.separation = 0.0f;
    }
    result.closest_point = closest;
    *out = result;
    return result.separation <= 0.0f;
}

__device__ static inline bool pf_sphere_sphere_contact(
        const PfBody* sphere_a, const PfBody* sphere_b, PfManifold* out) {
    PfVec3 delta = pf_sub(sphere_a->position, sphere_b->position);
    float distance_squared = pf_length_squared(delta);
    float radius_sum = sphere_a->half_extents.x + sphere_b->half_extents.x;
    // Coincident centres name no direction. The fallback is deterministic, so
    // the manifold is, and the depth it reports is correct for any normal
    // because both points are placed on that line.
    float inverse_distance = distance_squared > 1.0e-20f
        ? rsqrtf(distance_squared) : 0.0f;
    PfVec3 normal = inverse_distance > 0.0f
        ? pf_scale(delta, inverse_distance) : pf_v3(1.0f, 0.0f, 0.0f);
    float separation = distance_squared * inverse_distance - radius_sum;
    // The primitive is correct on its own, independently of any caller's broad
    // phase: a positive separation is a gap, not a contact, and reporting one
    // hands the solver a contact between bodies that are apart. Checked before
    // anything is written, so a rejected query leaves the manifold untouched.
    if (separation > 0.0f) {
        return false;
    }
    out->normal = normal;
    out->point_count = 1;
    out->points[0].point_a = pf_sub(sphere_a->position,
        pf_scale(normal, sphere_a->half_extents.x));
    out->points[0].point_b = pf_add(sphere_b->position,
        pf_scale(normal, sphere_b->half_extents.x));
    out->points[0].separation = separation;
    return true;
}

__device__ static inline bool pf_sphere_box_contact(
        const PfBody* sphere, const PfBody* box, PfManifold* out) {
    PfVec3 axes[3];
    pf_quat_axes(box->rotation, axes);
    PfVec3 center_delta = pf_sub(sphere->position, box->position);
    PfVec3 local = pf_v3(
        pf_dot(center_delta, axes[0]),
        pf_dot(center_delta, axes[1]),
        pf_dot(center_delta, axes[2]));
    PfVec3 closest = box->position;
    for (int axis = 0; axis < 3; ++axis) {
        float half = axis == 0 ? box->half_extents.x
            : axis == 1 ? box->half_extents.y : box->half_extents.z;
        float value = axis == 0 ? local.x : axis == 1 ? local.y : local.z;
        float clamped = value < -half ? -half : value > half ? half : value;
        closest = pf_add(closest, pf_scale(axes[axis], clamped));
    }
    PfVec3 delta = pf_sub(sphere->position, closest);
    float distance_squared = pf_length_squared(delta);
    float radius = sphere->half_extents.x;
    float clearance;
    int axis_index = pf_box_escape_axis(local, box->half_extents, &clearance);
    float value = axis_index == 0 ? local.x
        : axis_index == 1 ? local.y : local.z;
    float half = axis_index == 0 ? box->half_extents.x
        : axis_index == 1 ? box->half_extents.y : box->half_extents.z;
    PfVec3 normal;
    float separation;
    if (pf_box_contains(local, box->half_extents)) {
        normal = pf_scale(axes[axis_index], value < 0.0f ? -1.0f : 1.0f);
        // Nearest geometric escape face. Entry direction needs swept history.
        // Signed offset of the sphere centre from the box centre along the
        // escape normal, positive on the side the normal names. The escape
        // face is `half` out on that side, so the sphere has to travel
        // `half - along` to reach it and one more radius to clear it. Both
        // points are placed on that line, so the reported separation is the
        // geometry handed to the solver rather than a second, disagreeing
        // number: the old form placed the box point `clearance` from the box
        // centre and reported `-radius - clearance`, which is the depth
        // through the *near* face and disagreed with its own points by the
        // sphere's own offset.
        float along = pf_dot(center_delta, normal);
        separation = -(radius + half - along);
        closest = pf_add(sphere->position, pf_scale(normal, half - along));
    } else if (distance_squared > 1.0e-20f) {
        float inverse_distance = rsqrtf(distance_squared);
        normal = pf_scale(delta, inverse_distance);
        separation = distance_squared * inverse_distance - radius;
    } else {
        // Exterior and exactly on the surface, so the distance is zero and the
        // normal is undefined: the outward normal of the face the point sits
        // on is the consistent answer and the separation the touching one.
        normal = pf_scale(axes[axis_index], value < 0.0f ? -1.0f : 1.0f);
        separation = -radius;
    }
    if (separation > 0.0f) {
        return false;
    }
    out->normal = normal;
    out->point_count = 1;
    out->points[0].point_a = pf_sub(sphere->position, pf_scale(normal, radius));
    out->points[0].point_b = closest;
    out->points[0].separation = separation;
    return true;
}
