#pragma once
#include "constraint_support.cuh"

__device__ static inline int pf_constraint_row(const PfModel& m, PfWorkspace& w,
        bool clear_jacobian = true) {
    if (w.row_count >= w.row_capacity) { w.status = PF_CAPACITY; return -1; }
    int row = w.row_count++;
    w.rows[row] = {0,0,-1.0e30f,1.0e30f,0,0,-1,-1,0};
    // Dense point rows overwrite every entry; sparse rows need zero fill.
    if (clear_jacobian)
        for (int d = 0; d < m.nv; ++d) w.jacobian[row*m.nv+d] = 0;
    return row;
}
__device__ static inline void pf_constraint_points(const PfModel& m,
        PfWorkspace& w, int row, int a, int b, PfVec3 pa, PfVec3 pb, PfVec3 axis,
        bool angular = false) {
    for (int d = 0; d < m.nv; ++d) {
        PfVec3 ja = angular ? (a < 0 ? pf_v3(0,0,0) : pf_jacobian_entry(m,w,a,d,true))
            : pf_point_jacobian(m,w,a,pa,d);
        PfVec3 jb = angular ? (b < 0 ? pf_v3(0,0,0) : pf_jacobian_entry(m,w,b,d,true))
            : pf_point_jacobian(m,w,b,pb,d);
        w.jacobian[row*m.nv+d] = pf_dot(pf_sub(ja,jb),axis);
    }
}

// Implicit spring-damper constraint, normalized by effective mass. This is a
// constant-impedance model, not MuJoCo's nonlinear five-parameter solimp curve.
__device__ static inline void pf_constraint_prepare(const PfModel& m,
        PfWorkspace& w, int row, float error, float dt, float time_constant,
        float damping_ratio, float impedance = 1) {
    PfRow& r = w.rows[row];
    pf_articulated_solve(m,w,w.jacobian+row*m.nv,w.response+row*m.nv);
    float effective = pf_constraint_effective(m,w,row);
    float t = fmaxf(time_constant,2*dt);
    float k = 1/(t*t), damping = 2*damping_ratio/t;
    r.target = -k*error/(dt*k+damping);
    r.softness = effective*(1/(dt*(dt*k+damping)) + (1-impedance)/impedance);
    r.inverse_diagonal = effective > 1.0e-15f ? 1/(effective+r.softness) : 0;
    if (effective <= 1.0e-15f && fabsf(error)>1.0e-6f) w.status=PF_SINGULAR;
}
__device__ static inline void pf_constraint_hard(const PfModel& m,
        PfWorkspace& w, int row) {
    pf_articulated_solve(m,w,w.jacobian+row*m.nv,w.response+row*m.nv);
    float effective = pf_constraint_effective(m,w,row);
    w.rows[row].inverse_diagonal = effective > 1.0e-15f ? 1/effective : 0;
}
template<bool bounded>
__device__ static inline void pf_constraint_impulse_impl(const PfModel& m,
        PfState& s, PfWorkspace& w, int row, float impulse) {
    float delta = impulse-w.rows[row].impulse;
    w.rows[row].impulse = impulse;
    // Finite responses contribute nothing when the impulse is unchanged.
    if (delta == 0) return;
    int begin=bounded?w.rows[row].first_dof:0,end=bounded?pf_row_end(m,w.rows[row]):m.nv;
    for (int d=begin;d<end;++d)
        s.qvel[d] += delta*w.response[row*m.nv+d];
}
template<bool bounded>
__device__ static __noinline__ void pf_constraint_solve_impl(const PfModel& m,
        PfState& s, PfWorkspace& w, const PfStepOptions& o) {
    w.iterations = 0;
    for (int iteration = 0; iteration < o.iterations; ++iteration) {
        float largest = 0;
        for (int i = 0; i < w.row_count; ++i) {
            PfRow& r = w.rows[i];
            if (i+1 < w.row_count && w.rows[i+1].group_end == i) continue;
            if (r.group_end >= 0) {
                int first = r.group_end;
                PfRow& x = w.rows[first];
                float vx=0,vy=0,cross=0;
                int begin=bounded?(x.first_dof<r.first_dof?x.first_dof:r.first_dof):0;
                int xe=bounded?pf_row_end(m,x):m.nv,re=bounded?pf_row_end(m,r):m.nv,end=xe>re?xe:re;
                for (int d=begin;d<end;++d) {
                    vx+=w.jacobian[first*m.nv+d]*s.qvel[d];
                    vy+=w.jacobian[i*m.nv+d]*s.qvel[d];
                    cross+=w.jacobian[first*m.nv+d]*w.response[i*m.nv+d];
                }
                float xx=x.inverse_diagonal>0?1/x.inverse_diagonal:0;
                float yy=r.inverse_diagonal>0?1/r.inverse_diagonal:0;
                float eigen=0.5f*(xx+yy+sqrtf((xx-yy)*(xx-yy)+4*cross*cross));
                float step=eigen>1.0e-15f?1/eigen:0;
                float px=x.impulse+step*(x.target-vx-x.softness*x.impulse);
                float py=r.impulse+step*(r.target-vy-r.softness*r.impulse);
                float limit=r.friction*w.rows[r.normal].impulse;
                float length=sqrtf(px*px+py*py);
                if (length>limit && length>0) { px*=limit/length; py*=limit/length; }
                largest=fmaxf(largest,fmaxf(fabsf(px-x.impulse),fabsf(py-r.impulse)));
                pf_constraint_impulse_impl<bounded>(m,s,w,first,px);
                pf_constraint_impulse_impl<bounded>(m,s,w,i,py);
                continue;
            }
            float velocity = 0;
            int end=bounded?pf_row_end(m,r):m.nv;
            for (int d=bounded?r.first_dof:0;d<end;++d)
                velocity += w.jacobian[i*m.nv+d]*s.qvel[d];
            float limit = r.normal >= 0 ? r.friction*w.rows[r.normal].impulse : 0;
            float candidate = pf_clamp(r.impulse + r.inverse_diagonal*
                (r.target-velocity-r.softness*r.impulse),
                r.normal >= 0 ? -limit : r.lower, r.normal >= 0 ? limit : r.upper);
            largest = fmaxf(largest,fabsf(candidate-r.impulse));
            pf_constraint_impulse_impl<bounded>(m,s,w,i,candidate);
        }
        w.iterations = iteration+1;
        w.residual = largest; // maximum impulse update, not a force/KKT residual
        if (largest <= o.tolerance) break;
    }
}

__device__ static inline void pf_constraint_impulse(const PfModel& m,
        PfState& s,PfWorkspace& w,int row,float impulse) {
    pf_constraint_impulse_impl<true>(m,s,w,row,impulse);
}
__device__ static inline void pf_constraint_solve(const PfModel& m,
        PfState& s,PfWorkspace& w,const PfStepOptions& o) {
    int work=0;
    for (int row=0;row<w.row_count;++row)
        work+=pf_row_end(m,w.rows[row])-w.rows[row].first_dof;
    // Bounds are useful only when they remove substantial coordinate work.
    if (work<m.nv*w.row_count/2) pf_constraint_solve_impl<true>(m,s,w,o);
    else pf_constraint_solve_impl<false>(m,s,w,o);
}
