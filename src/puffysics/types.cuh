#pragma once

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PF_MAX_MANIFOLD_POINTS 4

typedef struct PfVec3 {
    float x;
    float y;
    float z;
} PfVec3;

typedef struct PfQuat {
    float w;
    float x;
    float y;
    float z;
} PfQuat;

typedef enum PfBodyMode {
    PF_STATIC = 0,
    PF_DYNAMIC = 1,
    PF_KINEMATIC = 2
} PfBodyMode;

typedef enum PfShapeKind {
    PF_BOX = 0,
    PF_SPHERE = 1,
    PF_CYLINDER = 2,
    PF_CAPSULE = 3
} PfShapeKind;
typedef struct PfShape PfShape;


typedef struct PfBody {
    int shape;
    int mode;
    PfVec3 half_extents;
    PfVec3 position;
    PfQuat rotation;
    PfVec3 linear_velocity;
    PfVec3 angular_velocity;
    PfVec3 force;
    PfVec3 torque;
    float inverse_mass;
    PfVec3 inverse_inertia_local;
    float friction;
    float restitution;
} PfBody;

typedef struct PfContactPoint {
    PfVec3 point_a;
    PfVec3 point_b;
    float separation;
} PfContactPoint;

typedef struct PfContactVelocity {
    float normal_mass, tangent_1_mass, tangent_2_mass;
    float normal_impulse, tangent_1_impulse, tangent_2_impulse;
    float restitution_bias;
} PfContactVelocity;

typedef struct PfManifold {
    int body_a;
    int body_b;
    PfVec3 normal;
    PfVec3 tangent_1;
    PfVec3 tangent_2;
    float static_friction;
    float dynamic_friction;
    float restitution;
    float normal_impulse;
    float tangent_1_impulse;
    float tangent_2_impulse;
    int converged;
    int point_count;
    PfVec3 inertia_a[3], inertia_b[3]; // World inverse-inertia columns, rebuilt before each velocity solve.
    PfContactPoint points[PF_MAX_MANIFOLD_POINTS];
    PfContactVelocity velocity[PF_MAX_MANIFOLD_POINTS];
} PfManifold;

typedef struct PfWorld {
    PfBody* bodies;
    PfManifold* manifolds;
    int body_count;
    PfShape* compound_shapes;
    int* compound_shape_counts;
    int manifold_capacity;
    int manifold_count;
    int env_index;
} PfWorld;

typedef enum PfJointType {
    PF_JOINT_REVOLUTE = 0
} PfJointType;

typedef struct PfJoint {
    int type;
    int parent_body;
    int child_body;
    PfVec3 axis;
    PfVec3 parent_anchor;
    PfVec3 child_anchor;
    float angle;
    float angular_velocity;
    float lower_limit;
    float upper_limit;
    float motor_target;
    float motor_max_torque;
    float motor_stiffness;
    float motor_damping;
    float armature;
    float damping;
} PfJoint;

typedef struct PfJointBatch {
    PfJoint* joints;
    int env_count;
    int joint_capacity;
    size_t joint_stride;
    float* scratch;
    size_t scratch_stride;
} PfJointBatch;

typedef struct PfJointWorld {
    PfJoint* joints;
    PfBody* bodies;
    int joint_count;
    int body_count;
    int joint_capacity;
    int body_capacity;
    int env_index;
    float* scratch;
    int scratch_capacity;
} PfJointWorld;

typedef struct PfSoftParticle {
    PfVec3 position;
    PfVec3 previous_position;
    PfVec3 velocity;
    float inverse_mass;
} PfSoftParticle;

typedef struct PfSoftDistanceConstraint {
    int particle_a;
    int particle_b;
    float rest_length;
    float compliance;
    float lambda;
} PfSoftDistanceConstraint;

typedef struct PfSoftVolumeConstraint {
    int p0;
    int p1;
    int p2;
    int p3;
    float rest_volume;
    float compliance;
    float lambda;
} PfSoftVolumeConstraint;

typedef struct PfSoftWorld {
    PfSoftParticle* particles;
    PfSoftDistanceConstraint* distance_constraints;
    PfSoftVolumeConstraint* volume_constraints;
    int particle_count;
    int distance_count;
    int volume_count;
    int particle_capacity;
    int distance_capacity;
    int volume_capacity;
    int env_index;
} PfSoftWorld;


typedef struct PfBatch {
    PfBody* bodies;
    PfManifold* manifolds;
    PfSoftParticle* soft_particles;
    PfSoftDistanceConstraint* soft_distance_constraints;
    PfSoftVolumeConstraint* soft_volume_constraints;
    int env_count;
    PfShape* compound_shapes;
    int* compound_shape_counts;
    int compound_shape_capacity;
    size_t compound_shape_stride;
    int body_capacity;
    int particle_capacity;
    int distance_capacity;
    int volume_capacity;
    size_t body_stride;
    size_t manifold_stride;
    size_t particle_stride;
    size_t distance_stride;
    size_t volume_stride;
} PfBatch;

__host__ __device__ static inline bool pf_number(float value) {
    return value == value && value < 1.0e30f && value > -1.0e30f;
}

__host__ __device__ static inline bool pf_vec_valid(PfVec3 value) {
    return pf_number(value.x) && pf_number(value.y) && pf_number(value.z);
}

__host__ __device__ static inline bool pf_quat_valid(PfQuat value) {
    return pf_number(value.w) && pf_number(value.x)
        && pf_number(value.y) && pf_number(value.z)
        && value.w * value.w + value.x * value.x
            + value.y * value.y + value.z * value.z > 1.0e-12f;
}

__host__ __device__ static inline PfQuat pf_quat_identity(void) {
    PfQuat result = {1.0f, 0.0f, 0.0f, 0.0f};
    return result;
}

/* pf_number admits |v| < 1e30, so the squared norm of a perfectly usable
 * quaternion can overflow float (and the squared norm of a small one can
 * underflow to zero). Scaling by the largest component keeps the squared
 * norm in range for every input pf_quat_valid accepts, and equivalent
 * scaled quaternions reduce to bit-identical results. */
__host__ __device__ static inline PfQuat pf_quat_normalize_robust(PfQuat q) {
    float scale = fmaxf(fmaxf(fabsf(q.w), fabsf(q.x)),
        fmaxf(fabsf(q.y), fabsf(q.z)));
    if (!(scale > 1.0e-20f)) {
        return pf_quat_identity();
    }
    float inverse = 1.0f / scale;
    float w = q.w * inverse, x = q.x * inverse;
    float y = q.y * inverse, z = q.z * inverse;
    float r = rsqrtf(w * w + x * x + y * y + z * z);
    PfQuat result = {w * r, x * r, y * r, z * r};
    return result;
}

/* A quaternion already within this relative tolerance of unit is stored
 * verbatim, so an already-normalised input round-trips bit for bit; anything
 * further out is renormalised. 1e-5 on the squared norm is a 5e-6 relative
 * scale error, three orders of magnitude below PF_POSITION_SLOP. */
#define PF_QUAT_UNIT_TOLERANCE 1.0e-5f
__host__ __device__ static inline PfQuat pf_quat_stored_unit(PfQuat q) {
    float norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm > 1.0f - PF_QUAT_UNIT_TOLERANCE
            && norm < 1.0f + PF_QUAT_UNIT_TOLERANCE) {
        return q;
    }
    return pf_quat_normalize_robust(q);
}

__host__ __device__ static inline bool pf_mode_valid(int mode) {
    return mode == PF_STATIC || mode == PF_DYNAMIC || mode == PF_KINEMATIC;
}

__host__ __device__ static inline bool pf_mode_is_dynamic(int mode) {
    return mode == PF_DYNAMIC;
}

__host__ __device__ static inline bool pf_mode_is_movable(int mode) {
    return mode == PF_DYNAMIC || mode == PF_KINEMATIC;
}

__host__ __device__ static inline bool pf_box(
        PfBody* out, PfBodyMode mode, PfVec3 half_extents, float mass,
        PfVec3 position, PfQuat rotation, float friction, float restitution) {
    if (out == NULL || !pf_mode_valid((int)mode) || !pf_vec_valid(position)
            || !pf_quat_valid(rotation) || !pf_vec_valid(half_extents)
            || half_extents.x <= 0.0f || half_extents.y <= 0.0f
            || half_extents.z <= 0.0f || !pf_number(mass)
            || (mode == PF_DYNAMIC && mass <= 0.0f)
            || (mode != PF_DYNAMIC && mass != 0.0f)
            || !pf_number(friction) || friction < 0.0f
            || !pf_number(restitution) || restitution < 0.0f
            || restitution > 1.0f) {
        return false;
    }
    PfBody body = {};
    body.shape = PF_BOX;
    body.mode = (int)mode;
    body.half_extents = half_extents;
    body.position = position;
    body.rotation = pf_quat_stored_unit(rotation);
    body.force = (PfVec3){0.0f, 0.0f, 0.0f};
    body.torque = (PfVec3){0.0f, 0.0f, 0.0f};
    body.friction = friction;
    body.restitution = restitution;
    if (mode == PF_DYNAMIC) {
        body.inverse_mass = 1.0f / mass;
        float x = half_extents.x * 2.0f;
        float y = half_extents.y * 2.0f;
        float z = half_extents.z * 2.0f;
        body.inverse_inertia_local = (PfVec3){
            12.0f / (mass * (y * y + z * z)),
            12.0f / (mass * (x * x + z * z)),
            12.0f / (mass * (x * x + y * y))
        };
    }
    *out = body;
    return true;
}

__host__ __device__ static inline bool pf_sphere(
        PfBody* out, PfBodyMode mode, float radius, float mass,
        PfVec3 position, float friction, float restitution) {
    if (out == NULL || !pf_mode_valid((int)mode) || !pf_vec_valid(position)
            || !pf_number(radius) || radius <= 0.0f || !pf_number(mass)
            || (mode == PF_DYNAMIC && mass <= 0.0f)
            || (mode != PF_DYNAMIC && mass != 0.0f)
            || !pf_number(friction) || friction < 0.0f
            || !pf_number(restitution) || restitution < 0.0f
            || restitution > 1.0f) {
        return false;
    }
    PfBody body = {};
    body.shape = PF_SPHERE;
    body.mode = (int)mode;
    body.half_extents = (PfVec3){radius, 0.0f, 0.0f};
    body.position = position;
    body.rotation = pf_quat_identity();
    body.force = (PfVec3){0.0f, 0.0f, 0.0f};
    body.torque = (PfVec3){0.0f, 0.0f, 0.0f};
    body.friction = friction;
    body.restitution = restitution;
    if (mode == PF_DYNAMIC) {
        body.inverse_mass = 1.0f / mass;
        body.inverse_inertia_local = (PfVec3){
            2.5f / (mass * radius * radius),
            2.5f / (mass * radius * radius),
            2.5f / (mass * radius * radius)
        };
    }
    *out = body;
    return true;
}
