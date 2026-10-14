#pragma once
#include "actuators.cuh"
#include "limits.cuh"
#include "equalities.cuh"
#include "model_collision.cuh"
#include "integrate.cuh"

__device__ static inline void pf_integrate_coordinates(const PfModel& m,PfState& s,float dt) {
    for (int i=0;i<m.link_count;++i) {
        const PfLink& l=m.links[i];
        if (l.joint==PF_FIXED || l.mimic) continue;
        float* q=s.qpos+l.qpos; const float* v=s.qvel+l.dof;
        if (l.joint==PF_SLIDE || l.joint==PF_HINGE) { q[0]+=dt*v[0]; continue; }
        if (l.joint==PF_FREE) {
            for (int k=0;k<3;++k) q[k]+=dt*v[k];
            q+=3; v+=3;
        }
        PfQuat rotation={q[0],q[1],q[2],q[3]};
        rotation=pf_quat_normalize(pf_quat_multiply(
            pf_orientation_delta(pf_v3(v[0],v[1],v[2]),dt),rotation));
        q[0]=rotation.w; q[1]=rotation.x; q[2]=rotation.y; q[3]=rotation.z;
    }
}

// One call advances exactly options.dt. Forces remain caller-owned; reset or
// overwrite them explicitly. Failed steps must be reset/restored before reuse.
// kinematics_ready requires an unchanged state/model since the last refresh.
__device__ static inline bool pf_articulated_prepare(const PfModel& m,
        PfState& s,PfWorkspace& w,const PfStepOptions& o,bool kinematics_ready=false) {
    w.status=PF_OK; w.row_count=0; w.contact_count=0; w.iterations=0; w.residual=0;
    if (!pf_number(o.dt) || o.dt<=0 || !pf_vec_valid(o.gravity)
            || !pf_number(o.tolerance) || o.tolerance<0 || o.iterations<=0
            || !pf_state_valid(m,s) || (!kinematics_ready && !pf_kinematics(m,s,w))) {
        w.status=PF_INVALID_STATE; return false;
    }
    pf_articulated_assemble(m,s,w,o.gravity,o.dt);
    if (!pf_actuators(m,s,w,o.dt)) { w.status=PF_INVALID_STATE; return false; }
    if (!pf_articulated_factor(m,w)) { w.status=PF_SINGULAR; return false; }
    pf_articulated_solve(m,w,w.rhs,w.solution);
    for (int d=0;d<m.nv;++d) s.qvel[d]+=o.dt*w.solution[d];
    return true;
}

// Consume caller-provided contacts after prepare; the constraint solver is unchanged.
__device__ static inline bool pf_articulated_finish(const PfModel& m,
        PfState& s,PfWorkspace& w,const PfStepOptions& o) {
    pf_contact_rows(m,w,o.dt);
    pf_limit_rows(m,s,w,o.dt);
    pf_equality_rows(m,s,w,o.dt);
    if (w.status!=PF_OK) return false;
    pf_constraint_solve(m,s,w,o);
    pf_integrate_coordinates(m,s,o.dt);
    if (!pf_state_valid(m,s) || !pf_kinematics(m,s,w)) {
        w.status=PF_INVALID_STATE; return false;
    }
    s.time+=o.dt;
    return true;
}

__device__ static inline bool pf_articulated_step(const PfModel& m,
        PfState& s,PfWorkspace& w,const PfStepOptions& o,bool kinematics_ready=false) {
    if (!pf_articulated_prepare(m,s,w,o,kinematics_ready)) return false;
    pf_model_collide(m,w);
    return pf_articulated_finish(m,s,w,o);
}
