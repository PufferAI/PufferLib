#pragma once
#include "constraints.cuh"

__device__ static inline void pf_limit_rows(const PfModel& m, const PfState& s,
        PfWorkspace& w, float dt) {
    for (int i = 0; i < m.link_count; ++i) {
        const PfLink& l = m.links[i];
        if (!l.limited) continue;
        float q = pf_joint_position(l,s.qpos), v = pf_joint_velocity(l,s.qvel);
        for (int side = 0; side < 2; ++side) {
            float sign = side ? -1 : 1;
            float gap = side ? l.upper-q : q-l.lower;
            if (gap > 0 && gap+dt*sign*v > 0) continue;
            int row = pf_constraint_row(m,w);
            if (row < 0) return;
            w.jacobian[row*m.nv+l.dof] = sign*pf_joint_scale(l);
            w.rows[row].lower = 0;
            if (gap >= 0) {
                pf_constraint_hard(m,w,row);
                w.rows[row].target = -gap/dt;
            } else pf_constraint_prepare(m,w,row,gap,dt,2*dt,1);
        }
    }
    for (int d = 0; d < m.nv; ++d) {
        if (m.dofs[d].friction_loss <= 0) continue;
        int row = pf_constraint_row(m,w);
        if (row < 0) return;
        w.jacobian[row*m.nv+d] = 1;
        w.rows[row].lower = -dt*m.dofs[d].friction_loss;
        w.rows[row].upper = dt*m.dofs[d].friction_loss;
        pf_constraint_hard(m,w,row);
    }
}
