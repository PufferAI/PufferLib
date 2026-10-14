#pragma once
#include "joint_coupling.cuh"

struct PfState {
    float *qpos, *qvel, *control, *activation, *applied;
    PfVec3 *force, *torque;      // world force at COM and world torque per link
    double time;
};
__device__ static inline bool pf_state_valid(const PfModel& m,const PfState& s) {
    if (!isfinite(s.time)) return false;
    for (int i=0;i<m.nq;++i) if (!pf_number(s.qpos[i])) return false;
    for (int i=0;i<m.nv;++i)
        if (!pf_number(s.qvel[i]) || !pf_number(s.applied[i])) return false;
    for (int i=0;i<m.actuator_count;++i)
        if (!pf_number(s.control[i]) || !pf_number(s.activation[i])) return false;
    for (int i=0;i<m.link_count;++i) {
        const PfLink& l=m.links[i];
        if (!pf_vec_valid(s.force[i]) || !pf_vec_valid(s.torque[i])) return false;
        if (l.joint==PF_BALL || l.joint==PF_FREE) {
            const float* q=s.qpos+l.qpos+(l.joint==PF_FREE?3:0);
            float norm=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
            if (!pf_number(norm) || fabsf(norm-1)>1.0e-4f) return false;
        }
    }
    return true;
}
struct PfLinkState {
    PfPose pose;
    PfVec3 velocity, angular_velocity, acceleration_bias, angular_bias;
};
struct PfRow {
    float target, softness, lower, upper, impulse, inverse_diagonal;
    int normal, group_end;      // friction row references its normal row
    float friction;
    int first_dof, end_dof; // Conservative support interval; end_dof==0 means dense.
};
struct PfContact {
    int geom_a, geom_b, first_row;
    PfVec3 point_a, point_b, normal;
    float separation;
    PfMaterial material;
};
struct PfWorkspace {
    PfLinkState* links;
    PfVec3 *linear_jacobian, *angular_jacobian; // At body origin; see compact_jacobian.
    float *matrix, *factor, *rhs, *solution, *diagonal;
    bool compact_matrix; // matrix/factor have model.matrix_size entries; otherwise nv*nv.
    bool sparse_jacobian; // Fixed topology; inactive entries initialized to zero.
    bool compact_jacobian; // Each array has model.dependency_count entries.
    PfRow* rows;
    float *jacobian, *response; // row_capacity * nv; response = effective_M^-1 J^T
    PfContact* contacts;
    int row_capacity, contact_capacity, row_count, contact_count, iterations;
    float residual;
    int status;
};
struct PfStepOptions {
    PfVec3 gravity;
    float dt, tolerance;
    int iterations;
};
__host__ __device__ static inline PfStepOptions pf_default_step_options() {
    return {pf_v3(0,-9.81f,0), 0.002f, 1.0e-5f, 40};
}
