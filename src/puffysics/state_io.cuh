#pragma once
#include "kinematics.cuh"

// Exact continuation state for this cold-start solver. Wrenches and controls
// are included; caches/Jacobians are rebuilt after restore. Time stays double.
__host__ __device__ static inline size_t pf_state_float_count(const PfModel& m) {
    return (size_t)m.nq+2*(size_t)m.nv+2*(size_t)m.actuator_count+6*(size_t)m.link_count;
}
__device__ static inline void pf_save_state(const PfModel& m,const PfState& s,float* out,double* time) {
    size_t k=0;
    for (int i=0;i<m.nq;++i) out[k++]=s.qpos[i];
    for (int i=0;i<m.nv;++i) { out[k++]=s.qvel[i]; out[k++]=s.applied[i]; }
    for (int i=0;i<m.actuator_count;++i) { out[k++]=s.control[i]; out[k++]=s.activation[i]; }
    for (int i=0;i<m.link_count;++i) {
        out[k++]=s.force[i].x; out[k++]=s.force[i].y; out[k++]=s.force[i].z;
        out[k++]=s.torque[i].x; out[k++]=s.torque[i].y; out[k++]=s.torque[i].z;
    }
    *time=s.time;
}
__device__ static inline void pf_restore_state(const PfModel& m,PfState& s,const float* in,double time) {
    size_t k=0;
    for (int i=0;i<m.nq;++i) s.qpos[i]=in[k++];
    for (int i=0;i<m.nv;++i) { s.qvel[i]=in[k++]; s.applied[i]=in[k++]; }
    for (int i=0;i<m.actuator_count;++i) { s.control[i]=in[k++]; s.activation[i]=in[k++]; }
    for (int i=0;i<m.link_count;++i) {
        s.force[i]={in[k],in[k+1],in[k+2]}; k+=3;
        s.torque[i]={in[k],in[k+1],in[k+2]}; k+=3;
    }
    s.time=time;
}
__device__ static inline bool pf_restore_state(const PfModel& m,PfState& s,
        PfWorkspace& w,const float* in,double time) {
    pf_restore_state(m,s,in,time);
    w.row_count=0; w.contact_count=0; w.iterations=0; w.residual=0;
    w.status=pf_state_valid(m,s) && pf_kinematics(m,s,w)?PF_OK:PF_INVALID_STATE;
    return w.status==PF_OK;
}
