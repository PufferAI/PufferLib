#pragma once
#include "shapes.cuh"
#include "collision_shapes.cuh"
#include "contact_rows.cuh"

// Convex assets contain outward-wound triangles with asset-local vertex IDs.
// Exact polyhedral SAT plus triangle clipping; quadratic edge work is intended
// for small offline-built hulls, not raw render meshes.
struct PfPoly {
    const PfVec3* vertices; const PfTriangle* faces;
    int vertex_count, face_count;
    PfPose pose;
};
__device__ static inline PfVec3 pf_poly_vertex(PfPoly p, int i) {
    return pf_pose_point(p.pose,p.vertices[i]);
}
__device__ static inline PfVec3 pf_poly_normal(PfPoly p, int i) {
    PfTriangle f = p.faces[i];
    return pf_normalize_or(pf_cross(pf_sub(pf_poly_vertex(p,f.b),pf_poly_vertex(p,f.a)),
        pf_sub(pf_poly_vertex(p,f.c),pf_poly_vertex(p,f.a))),pf_v3(0,1,0));
}
__device__ static inline void pf_poly_interval(PfPoly p, PfVec3 n, float& lo, float& hi) {
    lo = 1.0e30f; hi = -lo;
    for (int i = 0; i < p.vertex_count; ++i) {
        float x = pf_dot(pf_poly_vertex(p,i),n); lo = fminf(lo,x); hi = fmaxf(hi,x);
    }
}
__device__ static inline void pf_poly_edge(PfPoly p, int edge, PfVec3& a, PfVec3& b) {
    PfTriangle f = p.faces[edge/3];
    int ids[3] = {f.a,f.b,f.c};
    a = pf_poly_vertex(p,ids[edge%3]); b = pf_poly_vertex(p,ids[(edge+1)%3]);
}
__device__ static inline void pf_poly_bounds(PfPoly p, PfVec3& lo, PfVec3& hi) {
    lo = hi = pf_poly_vertex(p,0);
    for (int i = 1; i < p.vertex_count; ++i) {
        PfVec3 v = pf_poly_vertex(p,i);
        lo = pf_v3(fminf(lo.x,v.x),fminf(lo.y,v.y),fminf(lo.z,v.z));
        hi = pf_v3(fmaxf(hi.x,v.x),fmaxf(hi.y,v.y),fmaxf(hi.z,v.z));
    }
}
__device__ static inline void pf_poly_contact(const PfModel& m, PfWorkspace& w,
        int ga, int gb, PfPoly a, PfPoly b) {
    float margin = fmaxf(m.geoms[ga].material.margin,m.geoms[gb].material.margin);
    // Positive margins use the existing SAT-axis interpretation unchanged.
    if (margin == 0) {
        PfVec3 al,ah,bl,bh;
        pf_poly_bounds(a,al,ah); pf_poly_bounds(b,bl,bh);
        for (int axis = 0; axis < 3; ++axis) {
            float low_a = axis == 0 ? al.x : axis == 1 ? al.y : al.z;
            float high_a = axis == 0 ? ah.x : axis == 1 ? ah.y : ah.z;
            float low_b = axis == 0 ? bl.x : axis == 1 ? bl.y : bl.z;
            float high_b = axis == 0 ? bh.x : axis == 1 ? bh.y : bh.z;
            float scale = fmaxf(fmaxf(fabsf(low_a),fabsf(high_a)),
                fmaxf(fabsf(low_b),fabsf(high_b)));
            float padding = 1.0e-6f + 2.0e-6f*scale;
            if (low_a > high_b+padding || low_b > high_a+padding) return;
        }
    }
    float best = -1.0e30f;
    PfVec3 normal = pf_v3(0,1,0);
    int reference = 0, edge_a = -1, edge_b = -1;
    for (int source = 0; source < 2; ++source) {
        PfPoly p = source ? b : a;
        for (int f = 0; f < p.face_count; ++f) {
            PfVec3 n = pf_scale(pf_poly_normal(p,f),source ? 1 : -1);
            float al,ah,bl,bh; pf_poly_interval(a,n,al,ah); pf_poly_interval(b,n,bl,bh);
            float gap = al-bh;
            if (gap > margin) return;
            if (gap > best) { best = gap; normal = n; reference = source; }
        }
    }
    for (int ea = 0; ea < a.face_count*3; ++ea) {
        PfVec3 a0,a1; pf_poly_edge(a,ea,a0,a1);
        for (int eb = 0; eb < b.face_count*3; ++eb) {
            PfVec3 b0,b1; pf_poly_edge(b,eb,b0,b1);
            PfVec3 axis = pf_cross(pf_sub(a1,a0),pf_sub(b1,b0));
            if (pf_length_squared(axis) < 1.0e-16f) continue;
            axis = pf_normalize_or(axis,pf_v3(1,0,0));
            // The interval on -axis is [-hi,-lo]; retain the original sign order.
            float alo,ahi,blo,bhi;
            pf_poly_interval(a,axis,alo,ahi); pf_poly_interval(b,axis,blo,bhi);
            for (int sign = -1; sign <= 1; sign += 2) {
                PfVec3 n = pf_scale(axis,(float)sign);
                float al = sign < 0 ? -ahi : alo;
                float bh = sign < 0 ? -blo : bhi;
                float gap = al-bh;
                if (gap > margin) return;
                if (gap <= best+1.0e-6f || fabsf(pf_dot(a0,n)-al)>1.0e-5f
                        || fabsf(pf_dot(b0,n)-bh)>1.0e-5f) continue;
                best = gap; normal = n; edge_a = ea; edge_b = eb; reference = -1;
            }
        }
    }
    if (reference < 0) {
        PfVec3 a0,a1,b0,b1; pf_poly_edge(a,edge_a,a0,a1); pf_poly_edge(b,edge_b,b0,b1);
        float s,t; pf_closest_segment_segment(a0,a1,b0,b1,&s,&t);
        pf_contact_add(m,w,ga,gb,pf_add(a0,pf_scale(pf_sub(a1,a0),s)),
            pf_add(b0,pf_scale(pf_sub(b1,b0),t)),normal);
        return;
    }
    PfPoly ref = reference ? b : a, inc = reference ? a : b;
    PfVec3 outward = pf_scale(normal,reference ? 1 : -1);
    float rl,rh; pf_poly_interval(ref,outward,rl,rh);
    float min_dot = 1;
    for (int f = 0; f < inc.face_count; ++f)
        min_dot = fminf(min_dot,pf_dot(pf_poly_normal(inc,f),outward));
    // Triangulated coplanar faces form disjoint clipping regions. Processing
    // every region avoids requiring polygon reconstruction in the runtime.
    for (int rf = 0; rf < ref.face_count; ++rf) {
        if (pf_dot(pf_poly_normal(ref,rf),outward)<0.99999f) continue;
        for (int f = 0; f < inc.face_count; ++f) {
            if (pf_dot(pf_poly_normal(inc,f),outward)>min_dot+1.0e-5f) continue;
            PfTriangle face = inc.faces[f];
            PfVec3 p[8] = {pf_poly_vertex(inc,face.a),pf_poly_vertex(inc,face.b),pf_poly_vertex(inc,face.c)};
            PfVec3 scratch[8]; int count = 3;
            for (int e = 0; e < 3 && count; ++e) {
                PfVec3 x,y; pf_poly_edge(ref,rf*3+e,x,y);
                PfVec3 side = pf_normalize_or(pf_cross(pf_sub(y,x),outward),pf_v3(1,0,0));
                count = pf_clip_plane(p,count,scratch,side,pf_dot(side,x),false);
                for (int k = 0; k < count; ++k) p[k] = scratch[k];
            }
            for (int k = 0; k < count; ++k) {
                float gap = pf_dot(p[k],outward)-rh;
                PfVec3 projected = pf_sub(p[k],pf_scale(outward,gap));
                pf_contact_add(m,w,ga,gb,reference ? p[k] : projected,
                    reference ? projected : p[k],normal);
            }
        }
    }
}

__device__ static inline PfVec3 pf_triangle_closest(PfVec3 p, PfVec3 a, PfVec3 b, PfVec3 c) {
    PfVec3 ab = pf_sub(b,a), ac = pf_sub(c,a), ap = pf_sub(p,a);
    float d1=pf_dot(ab,ap), d2=pf_dot(ac,ap);
    if (d1<=0 && d2<=0) return a;
    PfVec3 bp=pf_sub(p,b); float d3=pf_dot(ab,bp), d4=pf_dot(ac,bp);
    if (d3>=0 && d4<=d3) return b;
    float vc=d1*d4-d3*d2;
    if (vc<=0 && d1>=0 && d3<=0) return pf_add(a,pf_scale(ab,d1/(d1-d3)));
    PfVec3 cp=pf_sub(p,c); float d5=pf_dot(ab,cp), d6=pf_dot(ac,cp);
    if (d6>=0 && d5<=d6) return c;
    float vb=d5*d2-d1*d6;
    if (vb<=0 && d2>=0 && d6<=0) return pf_add(a,pf_scale(ac,d2/(d2-d6)));
    float va=d3*d6-d5*d4;
    if (va<=0 && d4-d3>=0 && d5-d6>=0)
        return pf_add(b,pf_scale(pf_sub(c,b),(d4-d3)/(d4-d3+d5-d6)));
    float inverse=1/(va+vb+vc);
    return pf_add(a,pf_add(pf_scale(ab,vb*inverse),pf_scale(ac,vc*inverse)));
}
__device__ static inline void pf_poly_sphere_contact(const PfModel& m, PfWorkspace& w,
        int sphere, int hull, PfVec3 center, float radius, PfPoly poly) {
    float closest2=1.0e30f, max_plane=-1.0e30f;
    PfVec3 closest={}, face_normal={};
    for (int f=0; f<poly.face_count; ++f) {
        PfTriangle t=poly.faces[f]; PfVec3 a=pf_poly_vertex(poly,t.a);
        PfVec3 n=pf_poly_normal(poly,f);
        float plane=pf_dot(pf_sub(center,a),n);
        if (plane>max_plane) { max_plane=plane; face_normal=n; }
        PfVec3 p=pf_triangle_closest(center,a,pf_poly_vertex(poly,t.b),pf_poly_vertex(poly,t.c));
        float d=pf_length_squared(pf_sub(center,p));
        if (d<closest2) { closest2=d; closest=p; }
    }
    PfVec3 n;
    if (max_plane<=0) { n=face_normal; closest=pf_sub(center,pf_scale(n,max_plane)); }
    else n=pf_normalize_or(pf_sub(center,closest),face_normal);
    pf_contact_add(m,w,sphere,hull,pf_sub(center,pf_scale(n,radius)),closest,n);
}
