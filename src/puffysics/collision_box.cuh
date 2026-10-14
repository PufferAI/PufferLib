#pragma once

#include <float.h>

#include "math.cuh"

typedef struct PfBoxQuery {
    PfVec3 normal;
    float separation;
    int feature;
} PfBoxQuery;

// Discrete SAT uses geometry alone. Recovering an entry face requires a
// previous transform or swept query; current velocity cannot establish it.
__device__ static inline void pf_box_query_axis(
        PfVec3 axis, float separation, int feature,
        PfBoxQuery* query) {
    // Face features are tested before cross features, so a near-tie keeps the
    // face: its manifold retains the whole clipped contact patch instead of
    // collapsing to one edge point.
    float tolerance = 1.0e-4f * (1.0f + fabsf(query->separation));
    if (separation > query->separation + tolerance) {
        query->normal = axis;
        query->separation = separation;
        query->feature = feature;
    }
}

__device__ static inline float pf_box_half(
        const PfBody* body, int axis) {
    return axis == 0 ? body->half_extents.x
        : axis == 1 ? body->half_extents.y : body->half_extents.z;
}

__device__ static inline bool pf_box_sat(
        const PfBody* a, const PfBody* b, const PfVec3 axes_a[3],
        const PfVec3 axes_b[3], PfBoxQuery* query) {
    PfVec3 center_delta = pf_sub(b->position, a->position);
    // Row i of the relative matrix is built only when A-face axis i is tested;
    // the B and cross stages read the rows, so no dot product is evaluated
    // twice or before the rejection that makes it useless.
    float translation[3] = {};
    float relative[3][3] = {};
    float abs_relative[3][3] = {};
    query->normal = axes_a[0];
    query->separation = -FLT_MAX;
    query->feature = 0;
    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        translation[axis] = pf_dot(center_delta, axes_a[axis]);
        #pragma unroll
        for (int axis_b = 0; axis_b < 3; ++axis_b) {
            relative[axis][axis_b] = pf_dot(axes_a[axis], axes_b[axis_b]);
            abs_relative[axis][axis_b] = fabsf(relative[axis][axis_b]);
        }
        float radius_a = pf_box_half(a, axis);
        float radius_b = abs_relative[axis][0] * b->half_extents.x
            + abs_relative[axis][1] * b->half_extents.y
            + abs_relative[axis][2] * b->half_extents.z;
        float sign = translation[axis] >= 0.0f ? -1.0f : 1.0f;
        float separation = fabsf(translation[axis]) - radius_a - radius_b;
        if (separation > 0.0f) {
            return false;
        }
        PfVec3 normal = pf_scale(axes_a[axis], sign);
        pf_box_query_axis(normal, separation, axis, query);
    }
    #pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        float projection = translation[0] * relative[0][axis]
            + translation[1] * relative[1][axis]
            + translation[2] * relative[2][axis];
        float radius_b = pf_box_half(b, axis);
        float radius_a = abs_relative[0][axis] * a->half_extents.x
            + abs_relative[1][axis] * a->half_extents.y
            + abs_relative[2][axis] * a->half_extents.z;
        float sign = projection >= 0.0f ? -1.0f : 1.0f;
        float separation = fabsf(projection) - radius_a - radius_b;
        if (separation > 0.0f) {
            return false;
        }
        PfVec3 normal = pf_scale(axes_b[axis], sign);
        pf_box_query_axis(normal, separation, 3 + axis, query);
    }
    #pragma unroll
    for (int axis_a = 0; axis_a < 3; ++axis_a) {
        int next_a = (axis_a + 1) % 3;
        int previous_a = (axis_a + 2) % 3;
        #pragma unroll
        for (int axis_b = 0; axis_b < 3; ++axis_b) {
            float length_squared = 1.0f
                - abs_relative[axis_a][axis_b] * abs_relative[axis_a][axis_b];
            // `length_squared` is `1 - dot^2`, which for exact unit axes is
            // `|a x b|^2 = sin^2(theta)`. It is not a cross-product length and
            // it is not computed as one: it is a cancelling subtraction near
            // one, so its absolute error is set by the roundoff in `pf_dot`
            // and not by any real angle. `pf_quat_axes` is called once per
            // body and nvcc contracts the rotate and dot expressions
            // differently at the two call sites, so the two nominally
            // identical axis triples are not bit-identical. Measured over
            // 2048 random quaternions with provably identical rotations
            // (`docs/puffysics_contact_defects.md`): `|dot - 1|` reaches
            // 8.94e-07, about 7.5 ulp of 1.0, while the true
            // `|pf_cross(a, b)|` is 0. `1 - dot^2` therefore claims
            // `|a x b|^2` up to 2 x 8.94e-07 = 1.79e-06, exactly 15 ulp of
            // 1.0; a wider 4096-quaternion sweep gives 2.38e-06.
            //
            // That is the floor of the quantity being thresholded, so the
            // threshold has to sit above it or the test admits every axis the
            // arithmetic can fabricate. `1.0e-5f` is the smallest power of ten
            // clearing the measured maximum with margin (a factor of 4 over
            // 2.38e-06); it rejects cross axes within sin(theta) < 3.2e-03,
            // about 0.18 degrees, of parallel. Behaviour confirms the floor:
            // over 131072 random near-parallel overlapping box pairs scored
            // against a double-precision reference SAT, 514 contacts are
            // missed at 1e-07 or below, 236 at 2e-07, 12 at 5e-07 and 0 at
            // 1e-06 and above, while 131072 clearly separated near-parallel
            // pairs give 0 false contacts at every threshold from 1e-16 up.
            if (length_squared <= 1.0e-5f) {
                continue;
            }
            float projection = translation[previous_a] * relative[next_a][axis_b]
                - translation[next_a] * relative[previous_a][axis_b];
            float radius_a = pf_box_half(a, next_a)
                    * abs_relative[previous_a][axis_b]
                + pf_box_half(a, previous_a) * abs_relative[next_a][axis_b];
            int next_b = (axis_b + 1) % 3;
            int previous_b = (axis_b + 2) % 3;
            float radius_b = pf_box_half(b, next_b)
                    * abs_relative[axis_a][previous_b]
                + pf_box_half(b, previous_b) * abs_relative[axis_a][next_b];
            float separation = fabsf(projection) - radius_a - radius_b;
            if (separation > 0.0f) {
                return false;
            }
            float inverse_length = rsqrtf(length_squared);
            PfVec3 axis = pf_scale(pf_cross(axes_a[axis_a], axes_b[axis_b]),
                inverse_length);
            pf_box_query_axis(projection >= 0.0f ? pf_scale(axis, -1.0f) : axis,
                separation * inverse_length, 6 + axis_a * 3 + axis_b,
                query);
        }
    }
    return true;
}

__device__ static inline PfVec3 pf_box_support(
        const PfBody* body, const PfVec3 axes[3], PfVec3 direction) {
    PfVec3 result = body->position;
    for (int axis = 0; axis < 3; ++axis) {
        float half = axis == 0 ? body->half_extents.x
            : axis == 1 ? body->half_extents.y : body->half_extents.z;
        float sign = pf_dot(axes[axis], direction) >= 0.0f ? 1.0f : -1.0f;
        result = pf_add(result, pf_scale(axes[axis], half * sign));
    }
    return result;
}
