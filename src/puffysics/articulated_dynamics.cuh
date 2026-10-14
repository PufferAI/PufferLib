#pragma once
#include "kinematics.cuh"
#include "matrix_storage.cuh"

__device__ static inline PfVec3 pf_link_inertia(const PfLink& link,
        const PfLinkState& state, PfVec3 value) {
    PfQuat rotation = pf_quat_multiply(state.pose.rotation, link.inertia_rotation);
    PfVec3 local = pf_quat_rotate(pf_quat_conjugate(rotation), value);
    local = pf_v3(local.x*link.inertia.x, local.y*link.inertia.y, local.z*link.inertia.z);
    return pf_quat_rotate(rotation, local);
}

// Assemble M and the smooth force residual. pf_mass_entry exposes diagnostics;
// factor receives M + dt*D + dt^2*K for implicit damping and scalar springs.
__device__ static inline void pf_articulated_assemble(const PfModel& m,
        const PfState& s, PfWorkspace& w, PfVec3 gravity, float dt) {
    int size=w.compact_matrix?m.matrix_size:m.nv*m.nv;
    for (int i=0;i<size;++i) w.matrix[i]=0;
    for (int r = 0; r < m.nv; ++r) {
        w.rhs[r] = s.applied[r] - m.dofs[r].damping*s.qvel[r];
        w.diagonal[r] = dt*m.dofs[r].damping;
        w.matrix[pf_matrix_index(m,w,r,r)]=m.dofs[r].armature;
    }
    for (int i = 0; i < m.link_count; ++i) {
        const PfLink& link = m.links[i];
        const PfLinkState& body = w.links[i];
        PfVec3 offset = pf_quat_rotate(body.pose.rotation, link.center);
        PfVec3 bias = pf_add(body.acceleration_bias,
            pf_add(pf_cross(body.angular_bias, offset),
                pf_cross(body.angular_velocity,pf_cross(body.angular_velocity,offset))));
        PfVec3 force = pf_add(s.force[i], pf_scale(pf_sub(gravity,bias),link.mass));
        PfVec3 torque = pf_sub(s.torque[i],
            pf_add(pf_link_inertia(link,body,body.angular_bias),
                pf_cross(body.angular_velocity,pf_link_inertia(link,body,body.angular_velocity))));
        int begin = m.link_dof_offsets ? m.link_dof_offsets[i] : 0;
        int end = m.link_dof_offsets ? m.link_dof_offsets[i+1] : m.nv;
        for (int ri = begin; ri < end; ++ri) {
            int r = m.link_dof_offsets ? m.link_dofs[ri] : ri;
            PfVec3 ar = w.angular_jacobian[w.compact_jacobian?ri:i*m.nv+r];
            PfVec3 lr = pf_add(w.linear_jacobian[w.compact_jacobian?ri:i*m.nv+r],pf_cross(ar,offset));
            PfVec3 weighted = pf_link_inertia(link,body,ar);
            w.rhs[r] += pf_dot(lr,force) + pf_dot(ar,torque);
            for (int ci = begin; ci <= ri; ++ci) {
                int c = m.link_dof_offsets ? m.link_dofs[ci] : ci;
                PfVec3 ac = w.angular_jacobian[w.compact_jacobian?ci:i*m.nv+c];
                PfVec3 lc = pf_add(w.linear_jacobian[w.compact_jacobian?ci:i*m.nv+c],pf_cross(ac,offset));
                float value = link.mass*pf_dot(lr,lc)+pf_dot(weighted,ac);
                w.matrix[pf_matrix_index(m,w,r,c)] += value;
                if (!w.compact_matrix && r != c) w.matrix[c*m.nv+r] += value;
            }
        }
        if (!link.mimic && (link.joint == PF_HINGE || link.joint == PF_SLIDE)) {
            int d = link.dof;
            float k = m.dofs[d].stiffness;
            w.rhs[d] -= k*(s.qpos[link.qpos]-m.dofs[d].spring_reference+dt*s.qvel[d]);
            w.diagonal[d] += dt*dt*k;
        }
    }
}

// Only lower-triangle entries inside mass blocks are stored in factor.
// Compact storage omits structural zeros and the symmetric upper triangle.
__device__ static inline bool pf_articulated_factor(const PfModel& m, PfWorkspace& w) {
    int blocks = m.block_count ? m.block_count : 1;
    for (int b=0;b<blocks;++b) {
        int begin=m.block_count?m.block_offsets[b]:0;
        int end=m.block_count?m.block_offsets[b+1]:m.nv;
        for (int ri=begin;ri<end;++ri) {
            int r=m.block_count?m.block_dofs[ri]:ri;
            for (int ci=begin;ci<=ri;++ci) {
                int c=m.block_count?m.block_dofs[ci]:ci;
                float x=w.matrix[pf_matrix_index(m,w,r,c)]+(r==c?w.diagonal[r]:0);
                for (int ki=begin;ki<ci;++ki) {
                    int k=m.block_count?m.block_dofs[ki]:ki;
                    x-=w.factor[pf_matrix_index(m,w,r,k)]*w.factor[pf_matrix_index(m,w,c,k)];
                }
                if (!pf_number(x) || (r==c && x<=1.0e-12f)) return false;
                w.factor[pf_matrix_index(m,w,r,c)]=r==c?sqrtf(x):x/w.factor[pf_matrix_index(m,w,c,c)];
            }
        }
    }
    return true;
}
__device__ static inline void pf_articulated_solve(const PfModel& m,
        const PfWorkspace& w, const float* rhs, float* out) {
    int blocks=m.block_count?m.block_count:1;
    for (int b=0;b<blocks;++b) {
        int begin=m.block_count?m.block_offsets[b]:0;
        int end=m.block_count?m.block_offsets[b+1]:m.nv;
        bool active=false;
        for (int ri=begin;ri<end;++ri) {
            int r=m.block_count?m.block_dofs[ri]:ri;
            active |= rhs[r]!=0;
        }
        if (!active) {
            for (int ri=begin;ri<end;++ri) out[m.block_count?m.block_dofs[ri]:ri]=0;
            continue;
        }
        for (int ri=begin;ri<end;++ri) {
            int r=m.block_count?m.block_dofs[ri]:ri;
            float x=rhs[r];
            for (int ci=begin;ci<ri;++ci) {
                int c=m.block_count?m.block_dofs[ci]:ci;
                x-=w.factor[pf_matrix_index(m,w,r,c)]*out[c];
            }
            out[r]=x/w.factor[pf_matrix_index(m,w,r,r)];
        }
        for (int ri=end-1;ri>=begin;--ri) {
            int r=m.block_count?m.block_dofs[ri]:ri;
            float x=out[r];
            for (int ci=ri+1;ci<end;++ci) {
                int c=m.block_count?m.block_dofs[ci]:ci;
                x-=w.factor[pf_matrix_index(m,w,c,r)]*out[c];
            }
            out[r]=x/w.factor[pf_matrix_index(m,w,r,r)];
        }
    }
}
