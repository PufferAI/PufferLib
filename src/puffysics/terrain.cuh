#pragma once
#include "convex_collision.cuh"

__device__ static inline PfPose pf_geom_pose(const PfModel& m,
        const PfWorkspace& w, int geom) {
    const PfGeom& g=m.geoms[geom];
    return g.link<0 ? g.local : pf_pose_compose(w.links[g.link].pose,g.local);
}
__device__ static inline PfShapeWorld pf_geom_shape(const PfModel& m,
        const PfWorkspace& w, int geom) {
    const PfGeom& g=m.geoms[geom]; PfPose pose=pf_geom_pose(m,w,geom);
    PfVec3 velocity=pf_v3(0,0,0);
    if (g.link>=0) {
        const PfLinkState& b=w.links[g.link];
        velocity=pf_add(b.velocity,pf_cross(b.angular_velocity,pf_sub(pose.position,b.pose.position)));
    }
    return {(PfShapeKind)g.kind,pose.position,pose.rotation,g.size,velocity};
}
__device__ static inline void pf_plane_contact(const PfModel& m, PfWorkspace& w,
        int object, int plane) {
    PfPose p=pf_geom_pose(m,w,plane), g=pf_geom_pose(m,w,object);
    PfVec3 n=pf_quat_rotate(p.rotation,pf_v3(0,1,0));
    const PfGeom& geom=m.geoms[object];
    if (geom.kind==PF_GEOM_BOX || geom.kind==PF_GEOM_CONVEX) {
        PfConvex asset={};
        if (geom.kind==PF_GEOM_CONVEX) asset=m.convexes[geom.asset];
        int count=geom.kind==PF_GEOM_BOX ? 8 : asset.vertex_count;
        for (int i=0; i<count; ++i) {
            PfVec3 local=geom.kind==PF_GEOM_BOX
                ? pf_v3((i&1)?geom.size.x:-geom.size.x,(i&2)?geom.size.y:-geom.size.y,
                    (i&4)?geom.size.z:-geom.size.z) : m.vertices[asset.vertex_start+i];
            PfVec3 a=pf_pose_point(g,local);
            pf_contact_add(m,w,object,plane,a,pf_sub(a,pf_scale(n,pf_dot(pf_sub(a,p.position),n))),n);
        }
    } else {
        PfShapeWorld shape=pf_geom_shape(m,w,object);
        PfVec3 a=pf_shape_support(&shape,pf_scale(n,-1));
        // A flat cylinder cap needs a patch to resist rocking.
        PfVec3 axis=pf_shape_axis(&shape);
        bool cap=geom.kind==PF_GEOM_CYLINDER && fabsf(pf_dot(axis,n))>0.99999f;
        int count=cap ? 4 : 1;
        for (int i=0; i<count; ++i) {
            PfVec3 point=a;
            if (cap) point=pf_add(a,pf_quat_rotate(g.rotation,
                pf_v3((i==0?1:i==2?-1:0)*geom.size.x,0,(i==1?1:i==3?-1:0)*geom.size.x)));
            pf_contact_add(m,w,object,plane,point,
                pf_sub(point,pf_scale(n,pf_dot(pf_sub(point,p.position),n))),n);
        }
    }
}

// Sphere-heightfield contact against both top triangles in intersected cells.
// This first terrain path is one-sided and does not represent side/base walls.
// Other shape-heightfield combinations are rejected at model compilation.
__device__ static inline void pf_heightfield_contact(const PfModel& m,
        PfWorkspace& w, int sphere, int terrain) {
    const PfGeom& g=m.geoms[terrain]; const PfHeightfield& h=m.heightfields[g.asset];
    PfPose pose=pf_geom_pose(m,w,terrain);
    PfVec3 center=pf_geom_pose(m,w,sphere).position;
    PfVec3 local=pf_quat_rotate(pf_quat_conjugate(pose.rotation),pf_sub(center,pose.position));
    float radius=m.geoms[sphere].size.x;
    float reach=radius+fmaxf(g.material.margin,m.geoms[sphere].material.margin);
    if (local.x+reach<0 || local.z+reach<0 || local.x-reach>(h.nx-1)*h.dx
            || local.z-reach>(h.nz-1)*h.dz) return;
    int x0=(int)fmaxf(0,floorf((local.x-reach)/h.dx));
    int x1=(int)fminf(h.nx-2,floorf((local.x+reach)/h.dx));
    int z0=(int)fmaxf(0,floorf((local.z-reach)/h.dz));
    int z1=(int)fminf(h.nz-2,floorf((local.z+reach)/h.dz));
    for (int z=z0; z<=z1; ++z) for (int x=x0; x<=x1; ++x) {
        PfVec3 p[4];
        for (int k=0;k<4;++k) {
            int xx=x+(k&1), zz=z+(k>>1);
            p[k]=pf_v3(xx*h.dx,m.heights[h.start+zz*h.nx+xx],zz*h.dz);
        }
        for (int t=0;t<2;++t) {
            PfVec3 a=p[t?3:0], b=p[t?1:2], c=p[t?2:1];
            PfVec3 nearest=pf_triangle_closest(local,a,b,c);
            PfVec3 top=pf_normalize_or(pf_cross(pf_sub(b,a),pf_sub(c,a)),pf_v3(0,1,0));
            PfVec3 delta=pf_sub(local,nearest);
            PfVec3 n=pf_normalize_or(delta,top);
            if (pf_dot(n,top)<0) n=pf_scale(n,-1);
            // Only recover below a triangle when the normal projection is
            // inside that triangle; don't turn distant outside edges into walls.
            if (pf_dot(delta,top)<0) {
                PfVec3 projected=pf_sub(local,pf_scale(top,pf_dot(pf_sub(local,a),top)));
                if (pf_length_squared(pf_sub(projected,pf_triangle_closest(projected,a,b,c)))>1.0e-10f) continue;
                nearest=projected; n=top;
            }
            n=pf_quat_rotate(pose.rotation,n);
            pf_contact_add(m,w,sphere,terrain,pf_sub(center,pf_scale(n,radius)),
                pf_pose_point(pose,nearest),n);
        }
    }
}
