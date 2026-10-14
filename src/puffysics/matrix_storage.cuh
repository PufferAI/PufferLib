#pragma once
#include "state.cuh"

// Internal index: r >= c, both coordinates in the same mass block.
__device__ static inline int pf_matrix_index(const PfModel& m,
        const PfWorkspace& w, int r, int c) {
    return w.compact_matrix ? m.matrix_offsets[r]+m.dof_local[c] : r*m.nv+c;
}

// Diagnostic access independent of allocation layout, including structural zeros.
__device__ static inline float pf_mass_entry(const PfModel& m,
        const PfWorkspace& w, int r, int c) {
    if (!w.compact_matrix) return w.matrix[r*m.nv+c];
    if (m.dof_blocks[r]!=m.dof_blocks[c]) return 0;
    if (r<c) { int swap=r; r=c; c=swap; }
    return w.matrix[pf_matrix_index(m,w,r,c)];
}
__device__ static inline void pf_mass_dense(const PfModel& m,
        const PfWorkspace& w, float* out) {
    for (int r=0;r<m.nv;++r) for (int c=0;c<m.nv;++c)
        out[r*m.nv+c]=pf_mass_entry(m,w,r,c);
}
