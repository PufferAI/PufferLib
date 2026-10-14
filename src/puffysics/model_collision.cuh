#pragma once
#include "terrain.cuh"

__device__ static inline PfPoly pf_geom_poly(const PfModel& m,
        const PfWorkspace& w, int geom, PfVec3 box[8], PfTriangle faces[12]) {
    const PfGeom& g=m.geoms[geom];
    if (g.kind==PF_GEOM_CONVEX) {
        PfConvex c=m.convexes[g.asset];
        return {m.vertices+c.vertex_start,m.faces+c.face_start,c.vertex_count,c.face_count,pf_geom_pose(m,w,geom)};
    }
    for (int i=0;i<8;++i) box[i]=pf_v3((i&1)?g.size.x:-g.size.x,
        (i&2)?g.size.y:-g.size.y,(i&4)?g.size.z:-g.size.z);
    const PfTriangle triangles[12]={{0,2,3},{0,3,1},{4,5,7},{4,7,6},
        {0,1,5},{0,5,4},{2,6,7},{2,7,3},{0,4,6},{0,6,2},{1,3,7},{1,7,5}};
    for (int i=0;i<12;++i) faces[i]=triangles[i];
    return {box,faces,8,12,pf_geom_pose(m,w,geom)};
}
__device__ static inline void pf_model_collide(const PfModel& m, PfWorkspace& w) {
    w.contact_count=0;
    for (int i=0;i<m.pair_count && w.status==PF_OK;++i) {
        int a=m.pairs[i].a,b=m.pairs[i].b;
        int ka=m.geoms[a].kind,kb=m.geoms[b].kind;
        if (ka==PF_GEOM_PLANE || kb==PF_GEOM_PLANE) {
            pf_plane_contact(m,w,ka==PF_GEOM_PLANE?b:a,ka==PF_GEOM_PLANE?a:b); continue;
        }
        if (ka==PF_GEOM_HEIGHTFIELD || kb==PF_GEOM_HEIGHTFIELD) {
            pf_heightfield_contact(m,w,ka==PF_GEOM_HEIGHTFIELD?b:a,ka==PF_GEOM_HEIGHTFIELD?a:b); continue;
        }
        if (ka==PF_GEOM_CONVEX || kb==PF_GEOM_CONVEX) {
            PfVec3 va[8],vb[8]; PfTriangle fa[12],fb[12];
            if (ka==PF_GEOM_SPHERE || kb==PF_GEOM_SPHERE) {
                int sphere=ka==PF_GEOM_SPHERE?a:b,poly=sphere==a?b:a;
                pf_poly_sphere_contact(m,w,sphere,poly,pf_geom_pose(m,w,sphere).position,
                    m.geoms[sphere].size.x,pf_geom_poly(m,w,poly,va,fa));
            } else pf_poly_contact(m,w,a,b,pf_geom_poly(m,w,a,va,fa),pf_geom_poly(m,w,b,vb,fb));
            continue;
        }
        PfShapeWorld sa=pf_geom_shape(m,w,a),sb=pf_geom_shape(m,w,b);
        float radius=pf_shape_radius(&sa)+pf_shape_radius(&sb);
        if (pf_length_squared(pf_sub(sa.center,sb.center))>radius*radius) continue;
        PfManifold manifold={};
        if (pf_shape_contact(&sa,&sb,&manifold))
            for (int p=0;p<manifold.point_count;++p)
                pf_contact_add(m,w,a,b,manifold.points[p].point_a,
                    manifold.points[p].point_b,manifold.normal);
    }
}
