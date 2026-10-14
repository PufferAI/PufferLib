#pragma once
#include "constraints.cuh"

__host__ __device__ static inline PfVec3 pf_rotation_error(PfQuat q) {
    q = pf_quat_normalize(q);
    if (q.w < 0) q = {-q.w,-q.x,-q.y,-q.z};
    PfVec3 v = pf_v3(q.x,q.y,q.z);
    float length = pf_length(v);
    return pf_scale(v,length > 1.0e-8f ? 2*atan2f(length,q.w)/length : 2);
}
__device__ static inline void pf_equality_rows(const PfModel& m,
        const PfState& s, PfWorkspace& w, float dt) {
    for (int i = 0; i < m.equality_count; ++i) {
        const PfEquality& e = m.equalities[i];
        if (e.kind == PF_COUPLE) {
            int row = pf_constraint_row(m,w);
            if (row < 0) return;
            const PfLink& a = m.links[e.a]; const PfLink& b = m.links[e.b];
            w.jacobian[row*m.nv+a.dof] += pf_joint_scale(a);
            w.jacobian[row*m.nv+b.dof] -= e.ratio*pf_joint_scale(b);
            float error = pf_joint_position(a,s.qpos)-e.ratio*pf_joint_position(b,s.qpos)-e.offset;
            pf_constraint_prepare(m,w,row,error,dt,e.time_constant,e.damping_ratio);
            continue;
        }
        PfPose a = w.links[e.a].pose;
        PfPose b = e.b < 0 ? pf_pose_identity() : w.links[e.b].pose;
        PfVec3 pa = pf_pose_point(a,e.anchor_a), pb = pf_pose_point(b,e.anchor_b);
        PfVec3 error = pf_sub(pa,pb);
        PfQuat desired = pf_quat_multiply(b.rotation,e.relative_rotation);
        PfVec3 rotation_error = pf_rotation_error(
            pf_quat_multiply(a.rotation,pf_quat_conjugate(desired)));
        for (int k = 0; k < (e.kind == PF_WELD ? 6 : 3); ++k) {
            int row = pf_constraint_row(m,w,false);
            if (row < 0) return;
            PfVec3 axis = pf_basis(k%3);
            pf_constraint_points(m,w,row,e.a,e.b,pa,pb,axis,k>=3);
            // Angular rows stabilize small orientation errors using the
            // relative angular velocity; large-error log Jacobian is omitted.
            pf_constraint_prepare(m,w,row,pf_dot(k<3 ? error : rotation_error,axis),
                dt,e.time_constant,e.damping_ratio);
        }
    }
}
