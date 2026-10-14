#pragma once
#include "model_collision.cuh"

struct PfRayHit { int geom; float distance; PfVec3 point,normal; };
__device__ static inline void pf_ray_candidate(PfRayHit& hit,int geom,float t,
        PfVec3 normal,PfPose pose,PfVec3 origin,PfVec3 direction) {
    if (t<0 || t>=hit.distance) return;
    hit={geom,t,pf_add(origin,pf_scale(direction,t)),pf_quat_rotate(pose.rotation,normal)};
}
__device__ static inline bool pf_ray_triangle(PfVec3 o,PfVec3 d,PfVec3 a,PfVec3 b,PfVec3 c,float& t) {
    PfVec3 e=pf_sub(b,a),f=pf_sub(c,a),p=pf_cross(d,f);
    float det=pf_dot(e,p); if (fabsf(det)<1.0e-9f) return false;
    PfVec3 s=pf_sub(o,a); float u=pf_dot(s,p)/det;
    if (u<0 || u>1) return false;
    PfVec3 q=pf_cross(s,e); float v=pf_dot(d,q)/det;
    if (v<0 || u+v>1) return false;
    t=pf_dot(f,q)/det; return t>=0;
}
// Unit world direction; hit is nearest nonnegative surface crossing, including
// exits for rays starting inside. mask filters geom.type; -2 excludes no link,
// -1 excludes world-attached geometry.
__device__ static inline bool pf_raycast(const PfModel& m,const PfWorkspace& w,
        PfVec3 origin,PfVec3 direction,float max_distance,unsigned int mask,
        PfRayHit* out,int excluded_link=-2) {
    if (!out || !pf_vec_valid(origin) || !pf_vec_valid(direction) || !pf_number(max_distance)
            || max_distance<=0 || fabsf(pf_length_squared(direction)-1)>1.0e-4f) return false;
    PfRayHit hit={-1,max_distance,{},{}};
    for (int i=0;i<m.geom_count;++i) {
        const PfGeom& g=m.geoms[i]; if (!(g.type&mask) || g.link==excluded_link) continue;
        PfPose pose=pf_geom_pose(m,w,i); PfQuat inv=pf_quat_conjugate(pose.rotation);
        PfVec3 o=pf_quat_rotate(inv,pf_sub(origin,pose.position)),d=pf_quat_rotate(inv,direction);
        if (g.kind==PF_GEOM_PLANE) {
            if (fabsf(d.y)>1.0e-9f) pf_ray_candidate(hit,i,-o.y/d.y,pf_v3(0,1,0),pose,origin,direction);
        } else if (g.kind==PF_GEOM_BOX || g.kind==PF_GEOM_CONVEX) {
            PfVec3 verts[8]; PfTriangle faces[12];
            PfPoly poly=pf_geom_poly(m,w,i,verts,faces); poly.pose=pf_pose_identity();
            float enter=-1.0e30f,exit=1.0e30f; PfVec3 en={},ex={}; bool valid=true;
            for (int f=0;f<poly.face_count;++f) {
                PfVec3 n=pf_poly_normal(poly,f),p=pf_poly_vertex(poly,poly.faces[f].a);
                float numerator=pf_dot(n,pf_sub(p,o)),denominator=pf_dot(n,d);
                if (fabsf(denominator)<1.0e-9f) { if (numerator<0) valid=false; continue; }
                float t=numerator/denominator;
                if (denominator<0 && t>enter) { enter=t; en=n; }
                if (denominator>0 && t<exit) { exit=t; ex=n; }
            }
            if (valid && enter<=exit) pf_ray_candidate(hit,i,enter>=0?enter:exit,enter>=0?en:ex,pose,origin,direction);
        } else if (g.kind==PF_GEOM_HEIGHTFIELD) {
            const PfHeightfield& h=m.heightfields[g.asset];
            for (int z=0;z<h.nz-1;++z) for (int x=0;x<h.nx-1;++x) {
                PfVec3 p[4];
                for (int k=0;k<4;++k) { int xx=x+(k&1),zz=z+(k>>1);
                    p[k]=pf_v3(xx*h.dx,m.heights[h.start+zz*h.nx+xx],zz*h.dz); }
                for (int k=0;k<2;++k) {
                    PfVec3 a=p[k?3:0],b=p[k?1:2],c=p[k?2:1]; float t;
                    if (pf_ray_triangle(o,d,a,b,c,t)) pf_ray_candidate(hit,i,t,
                        pf_normalize_or(pf_cross(pf_sub(b,a),pf_sub(c,a)),pf_v3(0,1,0)),pose,origin,direction);
                }
            }
        } else {
            float radius=g.size.x,height=g.kind==PF_GEOM_SPHERE?0:g.size.y;
            if (g.kind!=PF_GEOM_SPHERE) {
                float aa=d.x*d.x+d.z*d.z,bb=o.x*d.x+o.z*d.z;
                float cc=o.x*o.x+o.z*o.z-radius*radius,disc=bb*bb-aa*cc;
                if (aa>1.0e-12f && disc>=0) for (int sign=-1;sign<=1;sign+=2) {
                    float t=(-bb+sign*sqrtf(disc))/aa,y=o.y+t*d.y;
                    if (fabsf(y)<=height) pf_ray_candidate(hit,i,t,
                        pf_normalize_or(pf_v3(o.x+t*d.x,0,o.z+t*d.z),pf_v3(1,0,0)),pose,origin,direction);
                }
            }
            if (g.kind==PF_GEOM_CYLINDER) {
                if (fabsf(d.y)>1.0e-9f) for (int sign=-1;sign<=1;sign+=2) {
                    float t=(sign*height-o.y)/d.y,x=o.x+t*d.x,z=o.z+t*d.z;
                    if (x*x+z*z<=radius*radius) pf_ray_candidate(hit,i,t,pf_v3(0,sign,0),pose,origin,direction);
                }
            } else for (int end=-1;end<=1;end+=2) {
                PfVec3 center=pf_v3(0,end*height,0),relative=pf_sub(o,center);
                float b=pf_dot(relative,d),disc=b*b-pf_dot(relative,relative)+radius*radius;
                if (disc<0) continue;
                for (int sign=-1;sign<=1;sign+=2) {
                    float t=-b+sign*sqrtf(disc); PfVec3 p=pf_add(o,pf_scale(d,t));
                    if (g.kind==PF_GEOM_CAPSULE && end*p.y<height) continue;
                    pf_ray_candidate(hit,i,t,pf_normalize_or(pf_sub(p,center),pf_v3(1,0,0)),pose,origin,direction);
                }
                if (g.kind==PF_GEOM_SPHERE) break;
            }
        }
    }
    *out=hit; return hit.geom>=0;
}
