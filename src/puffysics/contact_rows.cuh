#pragma once
#include "contact_jacobian.cuh"

__device__ static inline PfMaterial pf_contact_material(PfMaterial a, PfMaterial b) {
    // Explicit native mixing policy; MJCF priority/solmix is not implemented.
    return {sqrtf(a.friction*b.friction),sqrtf(a.torsion*b.torsion),
        sqrtf(a.rolling*b.rolling),fmaxf(a.time_constant,b.time_constant),
        fmaxf(a.damping_ratio,b.damping_ratio),fminf(a.impedance,b.impedance),
        fmaxf(a.margin,b.margin),a.dimension > b.dimension ? a.dimension : b.dimension};
}
__device__ static inline void pf_contact_add(const PfModel& m, PfWorkspace& w,
        int a, int b, PfVec3 pa, PfVec3 pb, PfVec3 normal) {
    float separation = pf_dot(pf_sub(pa,pb),normal);
    PfMaterial material = pf_contact_material(m.geoms[a].material,m.geoms[b].material);
    if (separation > material.margin+1.0e-6f) return;
    // All points are solved separately. Remove duplicates from triangulated
    // coplanar faces without collapsing a support patch to its centroid.
    for (int i = w.contact_count-1; i >= 0; --i) {
        const PfContact& c = w.contacts[i];
        if (c.geom_a != a || c.geom_b != b) break;
        if (pf_length_squared(pf_sub(c.point_a,pa)) < 1.0e-12f
                && pf_dot(c.normal,normal) > 0.9999f) return;
    }
    if (w.contact_count >= w.contact_capacity) { w.status = PF_CAPACITY; return; }
    w.contacts[w.contact_count++] = {a,b,-1,pa,pb,normal,separation,material};
}
__device__ static inline void pf_contact_rows(const PfModel& m, PfWorkspace& w, float dt) {
    for (int i = 0; i < w.contact_count; ++i) {
        PfContact& c = w.contacts[i];
        int a = m.geoms[c.geom_a].link, b = m.geoms[c.geom_b].link;
        PfVec3 n = c.normal;
        PfVec3 t = pf_normalize_or(pf_cross(n,fabsf(n.y)<0.9f ? pf_v3(0,1,0) : pf_v3(1,0,0)),pf_v3(0,0,1));
        PfVec3 u = pf_cross(n,t);
        int count=c.material.dimension;
        if (count>w.row_capacity-w.row_count) count=w.row_capacity-w.row_count;
        pf_contact_jacobian(m,w,w.row_count,count,a,b,c.point_a,c.point_b,n,t,u);
        for (int k = 0; k < c.material.dimension; ++k) {
            int row = pf_constraint_row(m,w,false);
            if (row < 0) return;
            if (k == 0) {
                c.first_row = row;
                w.rows[row].lower = 0;
                if (c.separation > 0) {
                    pf_constraint_hard(m,w,row);
                    w.rows[row].target = -c.separation/dt;
                } else pf_constraint_prepare(m,w,row,c.separation,dt,
                    c.material.time_constant,c.material.damping_ratio,c.material.impedance);
            } else {
                pf_constraint_hard(m,w,row);
                w.rows[row].normal = c.first_row;
                w.rows[row].friction = k < 3 ? c.material.friction
                    : k == 3 ? c.material.torsion : c.material.rolling;
                if (k == 2 || k == 5) w.rows[row].group_end = row-1;
            }
        }
    }
}
