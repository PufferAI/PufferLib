#pragma once
#include "articulated_dynamics.cuh"

__device__ static inline int pf_row_end(const PfModel& m,const PfRow& r) {
    return r.end_dof ? r.end_dof : m.nv;
}
// Bound both J and M^-1 J. An interval preserves coordinate summation order,
// including interleaved blocks and rows coupling independent articulations.
__device__ static inline float pf_constraint_effective(const PfModel& m,
        PfWorkspace& w,int row) {
    float effective=0;
    int first=m.nv,end=m.nv;
    for (int d=0;d<m.nv;++d) {
        float j=w.jacobian[row*m.nv+d],a=w.response[row*m.nv+d];
        effective+=j*a;
        if (j!=0 || a!=0) { if (first==m.nv) first=d; end=d+1; }
    }
    w.rows[row].first_dof=first; w.rows[row].end_dof=end;
    return effective;
}
