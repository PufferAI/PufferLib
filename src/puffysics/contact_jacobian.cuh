#pragma once
#include "constraints.cuh"

// Reuse each point Jacobian across the contact's normal and tangent axes.
// count may be truncated by row capacity; only writable rows are touched.
__device__ static inline void pf_contact_jacobian(const PfModel& m,
        PfWorkspace& w, int first, int count, int a, int b, PfVec3 pa,
        PfVec3 pb, PfVec3 n, PfVec3 t, PfVec3 u) {
    if (count<=0) return;
    for (int d=0;d<m.nv;++d) {
        PfVec3 j=pf_sub(pf_point_jacobian(m,w,a,pa,d),
            pf_point_jacobian(m,w,b,pb,d));
        w.jacobian[first*m.nv+d]=pf_dot(j,n);
        if (count>1) w.jacobian[(first+1)*m.nv+d]=pf_dot(j,t);
        if (count>2) w.jacobian[(first+2)*m.nv+d]=pf_dot(j,u);
        if (count>3) {
            j=pf_sub(a<0?pf_v3(0,0,0):pf_jacobian_entry(m,w,a,d,true),
                b<0?pf_v3(0,0,0):pf_jacobian_entry(m,w,b,d,true));
            w.jacobian[(first+3)*m.nv+d]=pf_dot(j,n);
            if (count>4) w.jacobian[(first+4)*m.nv+d]=pf_dot(j,t);
            if (count>5) w.jacobian[(first+5)*m.nv+d]=pf_dot(j,u);
        }
    }
}
