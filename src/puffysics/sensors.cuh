#pragma once
#include "kinematics.cuh"

__device__ static inline PfPose pf_site_pose(const PfModel& m,const PfWorkspace& w,int site) {
    const PfSite& s=m.sites[site];
    return s.link<0 ? s.local : pf_pose_compose(w.links[s.link].pose,s.local);
}
__device__ static inline void pf_site_jacobian(const PfModel& m,const PfWorkspace& w,
        int site,PfVec3* linear,PfVec3* angular) {
    int link=m.sites[site].link; PfVec3 point=pf_site_pose(m,w,site).position;
    for (int d=0;d<m.nv;++d) {
        linear[d]=pf_point_jacobian(m,w,link,point,d);
        angular[d]=link<0 ? pf_v3(0,0,0) : pf_jacobian_entry(m,w,link,d,true);
    }
}
__device__ static inline PfVec3 pf_site_velocity(const PfModel& m,const PfWorkspace& w,int site) {
    int link=m.sites[site].link;
    if (link<0) return pf_v3(0,0,0);
    const PfLinkState& b=w.links[link];
    return pf_add(b.velocity,pf_cross(b.angular_velocity,
        pf_sub(pf_site_pose(m,w,site).position,b.pose.position)));
}
__device__ static inline PfVec3 pf_site_gyro(const PfModel& m,const PfWorkspace& w,int site) {
    int link=m.sites[site].link;
    return link<0 ? pf_v3(0,0,0) : pf_quat_rotate(pf_quat_conjugate(pf_site_pose(m,w,site).rotation),
        w.links[link].angular_velocity);
}
// qacc is supplied explicitly: the caller chooses instantaneous accelerations
// or a finite difference of generalized velocities over the last step.
__device__ static inline PfVec3 pf_site_accelerometer(const PfModel& m,const PfWorkspace& w,
        int site,const float* qacc,PfVec3 gravity) {
    PfPose pose=pf_site_pose(m,w,site); int link=m.sites[site].link;
    PfVec3 acceleration=pf_v3(0,0,0);
    if (link>=0) {
        const PfLinkState& b=w.links[link]; PfVec3 r=pf_sub(pose.position,b.pose.position);
        acceleration=pf_add(b.acceleration_bias,pf_add(pf_cross(b.angular_bias,r),
            pf_cross(b.angular_velocity,pf_cross(b.angular_velocity,r))));
        for (int d=0;d<m.nv;++d) acceleration=pf_add(acceleration,
            pf_scale(pf_point_jacobian(m,w,link,pose.position,d),qacc[d]));
    }
    return pf_quat_rotate(pf_quat_conjugate(pose.rotation),pf_sub(acceleration,gravity));
}
// Contact-only wrench, averaged over dt, about a supplied world point.
// This is not an internal joint reaction-force sensor.
__device__ static inline void pf_contact_wrench(const PfModel& m,const PfWorkspace& w,
        int link,PfVec3 about,float dt,PfVec3* force,PfVec3* torque) {
    *force=pf_v3(0,0,0); *torque=*force;
    if (!(dt>0)) return;
    for (int i=0;i<w.contact_count;++i) {
        const PfContact& c=w.contacts[i]; if (c.first_row<0) continue;
        bool a=m.geoms[c.geom_a].link==link,b=m.geoms[c.geom_b].link==link;
        if (!a && !b) continue;
        float sign=a?1:-1;
        PfVec3 n=c.normal;
        PfVec3 t=pf_normalize_or(pf_cross(n,fabsf(n.y)<0.9f?pf_v3(0,1,0):pf_v3(1,0,0)),pf_v3(0,0,1));
        PfVec3 u=pf_cross(n,t),f=pf_v3(0,0,0),tau=f;
        for (int k=0;k<c.material.dimension;++k) {
            PfVec3 axis=k==0 || k==3?n:k==1 || k==4?t:u;
            PfVec3 value=pf_scale(axis,sign*w.rows[c.first_row+k].impulse/dt);
            if (k<3) f=pf_add(f,value); else tau=pf_add(tau,value);
        }
        *force=pf_add(*force,f);
        *torque=pf_add(*torque,pf_add(tau,pf_cross(pf_sub(a?c.point_a:c.point_b,about),f)));
    }
}
