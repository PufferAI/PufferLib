#pragma once

#include "math.cuh"

// Immutable, topologically ordered model. Pointers address host memory while
// compiling/validating a model and device memory while stepping it.
enum PfJointKind { PF_FIXED, PF_HINGE, PF_SLIDE, PF_BALL, PF_FREE };
enum PfGeomKind { PF_GEOM_BOX, PF_GEOM_SPHERE, PF_GEOM_CYLINDER,
    PF_GEOM_CAPSULE, PF_GEOM_PLANE, PF_GEOM_CONVEX, PF_GEOM_HEIGHTFIELD };
enum PfActuatorKind { PF_MOTOR, PF_POSITION_SERVO, PF_VELOCITY_SERVO };
enum PfEqualityKind { PF_CONNECT, PF_WELD, PF_COUPLE };
enum PfStatus { PF_OK, PF_INVALID_MODEL, PF_INVALID_STATE, PF_CAPACITY,
    PF_SINGULAR, PF_UNSUPPORTED };

struct PfPose { PfVec3 position; PfQuat rotation; };
struct PfLink {
    int parent, joint, qpos, dof;
    PfPose rest;                 // body pose in parent at joint coordinate zero
    PfVec3 anchor, axis;         // body-local anchor and axis at zero
    float mass;
    PfVec3 center;               // center of mass in body coordinates
    PfQuat inertia_rotation;    // principal inertia frame in body coordinates
    PfVec3 inertia;              // principal moments, not inverse moments
    bool limited;
    float lower, upper;          // hinge/slide only
    bool mimic;                 // scalar joint shares an earlier owner's coordinate
    int source;                 // link index; ignored unless mimic is true
    float ratio, offset;        // physical joint coordinate = ratio*q_source + offset
};
struct PfDof {
    float armature, damping, stiffness, spring_reference, friction_loss;
};
struct PfMaterial {
    float friction, torsion, rolling;
    float time_constant, damping_ratio, impedance, margin;
    int dimension;              // 1, 3, 4 or 6
};
struct PfGeom {
    int link;                   // -1: world
    int kind, asset;
    PfPose local;
    PfVec3 size;                // cylinder/capsule: radius, half-height, 0
    unsigned int type, affinity;
    PfMaterial material;
};
struct PfTriangle { int a, b, c; };
struct PfConvex {
    int vertex_start, vertex_count, face_start, face_count;
};
struct PfHeightfield {
    int start, nx, nz;
    float dx, dz;               // sample (x,z) is (x*dx,z*dz); height is local Y
};
struct PfPair { int a, b; };     // eligible geom pairs, compiled once
struct PfActuator {
    int link, kind;
    float gear, kp, kd, time_constant;
    float control_min, control_max, force_min, force_max;
};
struct PfEquality {
    int kind, a, b;             // link IDs; b may be world for connect/weld
    PfVec3 anchor_a, anchor_b;
    PfQuat relative_rotation;   // desired conjugate(R_b) * R_a for weld
    float ratio, offset;        // q_a - ratio*q_b - offset = 0 for coupling
    float time_constant, damping_ratio;
};
struct PfSite { int link; PfPose local; };
struct PfModel {
    const PfLink* links; const PfDof* dofs; const PfGeom* geoms;
    const PfPair* pairs; const PfActuator* actuators;
    const PfEquality* equalities; const PfSite* sites;
    const PfConvex* convexes; const PfVec3* vertices;
    const PfTriangle* faces; const PfHeightfield* heightfields;
    const float* heights; const float* initial_qpos;
    int link_count, nq, nv, geom_count, pair_count, actuator_count;
    int equality_count, site_count, convex_count, vertex_count, face_count;
    int heightfield_count, height_count;
    // Compiled coordinate dependencies and independent mass blocks (CSR).
    // IDs within each list are increasing; coordinates are never renumbered.
    const int *link_dof_offsets, *link_dofs, *block_offsets, *block_dofs;
    int dependency_count, block_count;
    const int *dof_blocks, *dof_local, *matrix_offsets;
    int matrix_size; // Sum of packed lower-triangle sizes of mass blocks.
    const int* jacobian_indices; // Dense logical index -> dependency index, or -1.
};

__host__ __device__ static inline int pf_joint_nq(int kind) {
    return kind == PF_FREE ? 7 : kind == PF_BALL ? 4
        : kind == PF_HINGE || kind == PF_SLIDE ? 1 : 0;
}
__host__ __device__ static inline int pf_joint_nv(int kind) {
    return kind == PF_FREE ? 6 : kind == PF_BALL ? 3
        : kind == PF_HINGE || kind == PF_SLIDE ? 1 : 0;
}
__host__ __device__ static inline PfPose pf_pose_identity() {
    return {pf_v3(0,0,0), pf_quat_identity()};
}
__host__ __device__ static inline PfPose pf_pose_compose(PfPose a, PfPose b) {
    return {pf_add(a.position, pf_quat_rotate(a.rotation, b.position)),
        pf_quat_multiply(a.rotation, b.rotation)};
}
__host__ __device__ static inline PfVec3 pf_pose_point(PfPose p, PfVec3 x) {
    return pf_add(p.position, pf_quat_rotate(p.rotation, x));
}
__host__ __device__ static inline PfVec3 pf_basis(int k) {
    return pf_v3(k == 0, k == 1, k == 2);
}
__host__ __device__ static inline float pf_clamp(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}
__host__ __device__ static inline PfMaterial pf_default_material() {
    return {0.7f, 0, 0, 0.02f, 1, 1, 0, 3};
}
