#pragma once

#include "math.cuh"
/* types.cuh owns the additive shape-kind enum. */

#define PF_MAX_SHAPES_PER_BODY 4

typedef struct PfShape {
    PfShapeKind kind;
    PfVec3 half_extents;
    PfVec3 position;
    PfQuat rotation;
} PfShape;

typedef struct PfMassProperties {
    PfVec3 center;
    float mass;
    float inertia[3][3];
    float off_diagonal_magnitude;
} PfMassProperties;

__host__ __device__ static inline bool pf_compound_shape_valid(
        const PfShape* shape) {
    if (shape == NULL || !pf_vec_valid(shape->half_extents)
            || !pf_vec_valid(shape->position) || !pf_quat_valid(shape->rotation)) return false;
    if (shape->kind == PF_BOX) {
        return shape->half_extents.x > 1.0e-6f
            && shape->half_extents.y > 1.0e-6f && shape->half_extents.z > 1.0e-6f;
    }
    if (shape->kind == PF_SPHERE) {
        return shape->half_extents.x > 1.0e-6f;
    }
    if (shape->kind == PF_CYLINDER || shape->kind == PF_CAPSULE) {
        return shape->half_extents.x > 1.0e-6f && shape->half_extents.y >= 0.0f;
    }
    return false;
}

__host__ __device__ static inline float pf_shape_volume(const PfShape* shape) {
    float x = shape->half_extents.x;
    float y = shape->half_extents.y;
    float z = shape->half_extents.z;
    if (shape->kind == PF_BOX) return 8.0f * x * y * z;
    if (shape->kind == PF_SPHERE) return 4.1887902047863905f * x * x * x;
    if (shape->kind == PF_CYLINDER) return 3.1415926535897932f * x * x * (2.0f * y);
    if (shape->kind == PF_CAPSULE) {
        return 3.1415926535897932f * x * x * (2.0f * y)
            + 4.1887902047863905f * x * x * x;
    }
    return 0.0f;
}

__host__ __device__ static inline void pf_shape_principal_inertia(
        const PfShape* shape, float mass, float inertia[3]) {
    float x = shape->half_extents.x;
    float y = shape->half_extents.y;
    float z = shape->half_extents.z;
    if (shape->kind == PF_BOX) {
        float width = 2.0f * x, height = 2.0f * y, depth = 2.0f * z;
        inertia[0] = mass * (height * height + depth * depth) / 12.0f;
        inertia[1] = mass * (width * width + depth * depth) / 12.0f;
        inertia[2] = mass * (width * width + height * height) / 12.0f;
    } else if (shape->kind == PF_SPHERE) {
        inertia[0] = inertia[1] = inertia[2] = 0.4f * mass * x * x;
    } else if (shape->kind == PF_CYLINDER) {
        float height = 2.0f * y;
        inertia[0] = inertia[2] = mass * (3.0f * x * x + height * height) / 12.0f;
        inertia[1] = 0.5f * mass * x * x;
    } else if (shape->kind == PF_CAPSULE) {
        /* r is half_extents.x, h the segment half-height half_extents.y, and
         * m_s the combined endcap mass. The hemispheres sit h away from the
         * centre, so their transverse moment carries h*h + 3*h*r/4 and their
         * axial moment stays 2*m_s*r*r/5. */
        float r = x;
        float cylinder_mass = mass * (2.0f * y) / (2.0f * y + 4.0f * x / 3.0f);
        float endcap_mass = mass - cylinder_mass;
        float height = 2.0f * y;
        inertia[0] = inertia[2] = cylinder_mass * (3.0f * r * r
            + height * height) / 12.0f + endcap_mass * (0.4f * r * r
            + y * y + 0.75f * y * r);
        inertia[1] = 0.5f * cylinder_mass * r * r
            + 0.4f * endcap_mass * r * r;
    } else {
        inertia[0] = inertia[1] = inertia[2] = 0.0f;
    }
}

__host__ __device__ static inline void pf_shape_add_tensor(
        float tensor[3][3], const PfQuat rotation, const float principal[3]) {
    PfVec3 axes[3];
    pf_quat_axes(rotation, axes);
    /* R * diag(principal) * R^T, with R columns equal to the shape axes. */
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            float value = 0.0f;
            for (int axis = 0; axis < 3; ++axis) {
                value += (row == 0 ? axes[axis].x : row == 1 ? axes[axis].y : axes[axis].z)
                    * principal[axis]
                    * (column == 0 ? axes[axis].x : column == 1 ? axes[axis].y : axes[axis].z);
            }
            tensor[row][column] += value;
        }
    }
}

/* Uniform density is an approximation; the authored body mass remains exact. */
/* ponytail: body dynamics retain a diagonal inverse inertia. Significant
 * off-diagonal terms are discarded; upgrade to a sidecar full tensor and
 * tensor-aware dynamics when target links require exact non-diagonal inertia. */
__host__ __device__ static inline bool pf_compound_mass_properties(
        const PfBody* body, const PfShape* shapes, int shape_count,
        PfMassProperties* out) {
    if (out == NULL || body == NULL || shapes == NULL || shape_count <= 0
            || shape_count > PF_MAX_SHAPES_PER_BODY || !pf_number(body->inverse_mass)
            || body->inverse_mass <= 0.0f) return false;
    float volumes[PF_MAX_SHAPES_PER_BODY] = {};
    float total_volume = 0.0f;
    for (int i = 0; i < shape_count; ++i) {
        if (!pf_compound_shape_valid(&shapes[i])) return false;
        volumes[i] = pf_shape_volume(&shapes[i]);
        if (!pf_number(volumes[i]) || volumes[i] <= 0.0f) return false;
        total_volume += volumes[i];
    }
    if (!pf_number(total_volume) || total_volume <= 0.0f) return false;
    float mass = 1.0f / body->inverse_mass;
    /* Uniform density makes a shape's mass share its volume share, so the
     * centre is the volume-weighted mean. Body-origin contract: the returned
     * centre and tensor are both about the body origin, which the caller must
     * make the centre of mass by shifting body position and subtracting the
     * centre from every shape position and joint anchor. pf_rek_reset is the
     * reference implementation of that contract. */
    PfVec3 center = pf_v3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < shape_count; ++i) {
        center = pf_add(center, pf_scale(shapes[i].position,
            volumes[i] / total_volume));
    }
    float tensor[3][3] = {};
    for (int i = 0; i < shape_count; ++i) {
        float shape_mass = mass * volumes[i] / total_volume;
        float principal[3];
        pf_shape_principal_inertia(&shapes[i], shape_mass, principal);
        pf_shape_add_tensor(tensor, shapes[i].rotation, principal);
        PfVec3 d = pf_sub(shapes[i].position, center);
        float dd = pf_length_squared(d);
        tensor[0][0] += shape_mass * (dd - d.x * d.x);
        tensor[1][1] += shape_mass * (dd - d.y * d.y);
        tensor[2][2] += shape_mass * (dd - d.z * d.z);
        tensor[0][1] -= shape_mass * d.x * d.y;
        tensor[0][2] -= shape_mass * d.x * d.z;
        tensor[1][2] -= shape_mass * d.y * d.z;
        tensor[1][0] = tensor[0][1];
        tensor[2][0] = tensor[0][2];
        tensor[2][1] = tensor[1][2];
    }
    float off_diagonal = sqrtf(tensor[0][1] * tensor[0][1]
        + tensor[0][2] * tensor[0][2] + tensor[1][2] * tensor[1][2]);
    PfMassProperties result = {};
    result.center = center;
    result.mass = mass;
    for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column)
        result.inertia[row][column] = tensor[row][column];
    result.off_diagonal_magnitude = off_diagonal;
    *out = result;
    return true;
}

__host__ __device__ static inline void pf_compound_apply_inverse_diagonal(
        PfBody* body, const PfMassProperties* properties) {
    if (body == NULL || properties == NULL || !pf_mode_is_dynamic(body->mode)) return;
    body->inverse_mass = 1.0f / properties->mass;
    body->inverse_inertia_local = pf_v3(
        properties->inertia[0][0] > 1.0e-12f ? 1.0f / properties->inertia[0][0] : 0.0f,
        properties->inertia[1][1] > 1.0e-12f ? 1.0f / properties->inertia[1][1] : 0.0f,
        properties->inertia[2][2] > 1.0e-12f ? 1.0f / properties->inertia[2][2] : 0.0f);
}
