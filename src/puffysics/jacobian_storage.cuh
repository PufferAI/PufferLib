#pragma once
#include "state.cuh"

// Logical access also supports structural zeros and legacy dense workspaces.
__device__ static inline PfVec3 pf_jacobian_entry(const PfModel& m,
        const PfWorkspace& w,int link,int dof,bool angular=false) {
    if (link<0) return pf_v3(0,0,0);
    int at=link*m.nv+dof;
    if (w.compact_jacobian) at=m.jacobian_indices[at];
    if (at<0) return pf_v3(0,0,0);
    return angular?w.angular_jacobian[at]:w.linear_jacobian[at];
}
