#pragma once
#include <vector>
#include <limits.h>
#include "model.cuh"
#include "model_topology.cuh"

// Offline storage only. The device sees PfModel's plain arrays.
struct PfModelStorage {
    PfTopologyStorage topology; // Derived; rebuilt on compile/cache load.
    std::vector<PfLink> links; std::vector<PfDof> dofs;
    std::vector<PfGeom> geoms; std::vector<PfPair> pairs, excluded_links;
    std::vector<PfActuator> actuators; std::vector<PfEquality> equalities;
    std::vector<PfSite> sites; std::vector<PfConvex> convexes;
    std::vector<PfVec3> vertices; std::vector<PfTriangle> faces;
    std::vector<PfHeightfield> heightfields; std::vector<float> heights, initial_qpos;
    bool exclude_parent_child = true;
    bool explicit_pairs = false; // pairs is an authoritative geom whitelist when true
};
static inline bool pf_pose_valid(PfPose p) {
    if (!pf_vec_valid(p.position) || !pf_quat_valid(p.rotation)) return false;
    PfQuat q=p.rotation;
    return fabsf(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z-1)<1.0e-5f;
}
static inline bool pf_nonnegative(float x) { return pf_number(x) && x>=0; }
static inline bool pf_material_valid(PfMaterial p) {
    return pf_nonnegative(p.friction) && pf_nonnegative(p.torsion) && pf_nonnegative(p.rolling)
        && pf_number(p.time_constant) && p.time_constant>0 && pf_nonnegative(p.damping_ratio)
        && pf_number(p.impedance) && p.impedance>0 && p.impedance<=1
        && pf_nonnegative(p.margin) && (p.dimension==1 || p.dimension==3 || p.dimension==4 || p.dimension==6);
}
static inline bool pf_geom_pair_supported(int a,int b) {
    if (a==PF_GEOM_HEIGHTFIELD || b==PF_GEOM_HEIGHTFIELD)
        return (a==PF_GEOM_HEIGHTFIELD?b:a)==PF_GEOM_SPHERE;
    if (a==PF_GEOM_PLANE || b==PF_GEOM_PLANE)
        return (a==PF_GEOM_PLANE?b:a)!=PF_GEOM_PLANE;
    if (a==PF_GEOM_CONVEX || b==PF_GEOM_CONVEX) {
        int other=a==PF_GEOM_CONVEX?b:a;
        return other==PF_GEOM_BOX || other==PF_GEOM_SPHERE || other==PF_GEOM_CONVEX;
    }
    return true;
}

// Returns a literal diagnostic, or nullptr on success. No silent conversion of
// unsupported joints, shape pairs, or spring models. Call before device upload.
static inline const char* pf_compile_model(PfModelStorage& h,PfModel& m) {
    m={};
    for (size_t n:{h.links.size(),h.geoms.size(),h.dofs.size(),h.actuators.size(),
            h.equalities.size(),h.sites.size(),h.convexes.size(),h.vertices.size(),
            h.faces.size(),h.heightfields.size(),h.heights.size(),h.initial_qpos.size(),h.pairs.size()})
        if (n>INT_MAX/3) return "model too large";
    int nq=0,nv=0;
    std::vector<int> root(h.links.size(),-1), welded(h.links.size(),-1);
    for (int i=0;i<(int)h.links.size();++i) {
        PfLink& l=h.links[i];
        if (l.parent < -1 || l.parent>=i || l.joint<PF_FIXED || l.joint>PF_FREE)
            return "links must be topological with a supported joint";
        if (l.joint==PF_FREE && l.parent!=-1) return "free joint must be a root";
        if (!pf_pose_valid(l.rest) || !pf_pose_valid({l.center,l.inertia_rotation})
                || !pf_vec_valid(l.anchor) || !pf_vec_valid(l.axis)
                || !pf_nonnegative(l.mass) || !pf_vec_valid(l.inertia)
                || l.inertia.x<0 || l.inertia.y<0 || l.inertia.z<0)
            return "invalid link pose or mass properties";
        if (l.mass>0 && (l.inertia.x<=0 || l.inertia.y<=0 || l.inertia.z<=0
                || l.inertia.x>l.inertia.y+l.inertia.z+1.0e-6f
                || l.inertia.y>l.inertia.x+l.inertia.z+1.0e-6f
                || l.inertia.z>l.inertia.x+l.inertia.y+1.0e-6f)) return "nonphysical inertia";
        if (l.mass==0 && pf_length_squared(l.inertia)!=0) return "massless frame has inertia";
        if ((l.joint==PF_HINGE || l.joint==PF_SLIDE)
                && fabsf(pf_length_squared(l.axis)-1)>1.0e-5f) return "joint axis must be unit length";
        if (l.limited && ((l.joint!=PF_HINGE && l.joint!=PF_SLIDE)
                || !pf_number(l.lower) || !pf_number(l.upper) || l.lower>=l.upper))
            return "limits require a hinge/slide with lower < upper";
        if (nq>INT_MAX-7 || nv>INT_MAX-6) return "coordinate count overflow";
        if (l.mimic) {
            if ((l.joint!=PF_SLIDE && l.joint!=PF_HINGE) || l.source<0 || l.source>=i
                    || h.links[l.source].mimic || h.links[l.source].joint!=l.joint
                    || !pf_number(l.ratio) || l.ratio==0 || !pf_number(l.offset))
                return "mimic requires an earlier independent joint of the same scalar kind";
            l.qpos=h.links[l.source].qpos; l.dof=h.links[l.source].dof;
        } else {
            l.qpos=nq; l.dof=nv; nq+=pf_joint_nq(l.joint); nv+=pf_joint_nv(l.joint);
        }
        root[i]=l.parent<0 ? (l.joint==PF_FIXED?-1:i) : root[l.parent];
        if (root[i]<0 && l.joint!=PF_FIXED) root[i]=i;
        welded[i]=l.joint!=PF_FIXED ? i : l.parent<0 ? -1 : welded[l.parent];
    }
    if (nv==0) return "model needs at least one degree of freedom";
    if ((size_t)nv*nv>INT_MAX || h.links.size()*(size_t)nv>INT_MAX)
        return "dense workspace exceeds native index range";
    if (h.dofs.empty()) h.dofs.resize(nv);
    if ((int)h.dofs.size()!=nv) return "dof parameter count mismatch";
    for (int i=0;i<(int)h.links.size();++i) {
        const PfLink& l=h.links[i];
        if (l.mimic) continue; // Passive parameters belong to the independent DOF.
        for (int k=0;k<pf_joint_nv(l.joint);++k) {
            const PfDof& d=h.dofs[l.dof+k];
            if (!pf_nonnegative(d.armature) || !pf_nonnegative(d.damping)
                    || !pf_nonnegative(d.stiffness) || !pf_number(d.spring_reference)
                    || !pf_nonnegative(d.friction_loss)) return "invalid passive dof parameters";
            if (d.stiffness>0 && l.joint!=PF_HINGE && l.joint!=PF_SLIDE)
                return "spring stiffness currently requires a scalar joint";
        }
    }
    if (h.initial_qpos.empty()) {
        h.initial_qpos.resize(nq);
        for (const PfLink& l:h.links) {
            if (l.joint==PF_BALL) h.initial_qpos[l.qpos]=1;
            if (l.joint==PF_FREE) {
                float* q=h.initial_qpos.data()+l.qpos;
                q[0]=l.rest.position.x; q[1]=l.rest.position.y; q[2]=l.rest.position.z;
                q[3]=l.rest.rotation.w; q[4]=l.rest.rotation.x;
                q[5]=l.rest.rotation.y; q[6]=l.rest.rotation.z;
            }
        }
    }
    if ((int)h.initial_qpos.size()!=nq) return "initial qpos count mismatch";
    for (float q:h.initial_qpos) if (!pf_number(q)) return "nonfinite initial qpos";
    for (const PfLink& l:h.links) if (l.joint==PF_BALL || l.joint==PF_FREE) {
        const float* q=h.initial_qpos.data()+l.qpos+(l.joint==PF_FREE?3:0);
        if (!pf_pose_valid({pf_v3(0,0,0),{q[0],q[1],q[2],q[3]}})) return "initial quaternion must be unit";
    }
    for (const PfConvex& c:h.convexes) {
        if (c.vertex_start<0 || c.vertex_count<4 || c.face_start<0 || c.face_count<4
                || (size_t)c.vertex_start+c.vertex_count>h.vertices.size()
                || (size_t)c.face_start+c.face_count>h.faces.size()) return "invalid convex asset range";
        for (int f=0;f<c.face_count;++f) {
            PfTriangle t=h.faces[c.face_start+f];
            if (t.a<0 || t.b<0 || t.c<0 || t.a>=c.vertex_count || t.b>=c.vertex_count || t.c>=c.vertex_count)
                return "convex face index out of range";
            PfVec3 a=h.vertices[c.vertex_start+t.a],b=h.vertices[c.vertex_start+t.b],v=h.vertices[c.vertex_start+t.c];
            PfVec3 n=pf_cross(pf_sub(b,a),pf_sub(v,a));
            if (!pf_vec_valid(a) || !pf_vec_valid(b) || !pf_vec_valid(v)
                    || !pf_vec_valid(n) || pf_length_squared(n)<1.0e-20f)
                return "degenerate convex face";
            n=pf_normalize_or(n,pf_v3(0,1,0));
            for (int k=0;k<c.vertex_count;++k)
                if (!pf_vec_valid(h.vertices[c.vertex_start+k])
                        || pf_dot(n,pf_sub(h.vertices[c.vertex_start+k],a))>1.0e-5f)
                    return "mesh must be convex and outward wound";
            int ids[3]={t.a,t.b,t.c};
            for (int e=0;e<3;++e) {
                int reverse=0,forward=0;
                for (int j=0;j<c.face_count;++j) {
                    PfTriangle other=h.faces[c.face_start+j]; int oi[3]={other.a,other.b,other.c};
                    for (int k=0;k<3;++k) {
                        reverse+=oi[k]==ids[(e+1)%3] && oi[(k+1)%3]==ids[e];
                        forward+=oi[k]==ids[e] && oi[(k+1)%3]==ids[(e+1)%3];
                    }
                }
                if (reverse!=1 || forward!=1) return "convex mesh is not a closed manifold";
            }
        }
    }
    for (const PfHeightfield& f:h.heightfields) {
        if (f.start<0 || f.nx<2 || f.nz<2 || !pf_number(f.dx) || f.dx<=0
                || !pf_number(f.dz) || f.dz<=0
                || (size_t)f.start+(size_t)f.nx*f.nz>h.heights.size()) return "invalid heightfield";
    }
    for (float x:h.heights) if (!pf_number(x)) return "invalid height sample";
    for (const PfGeom& g:h.geoms) {
        if (g.link < -1 || g.link>=(int)h.links.size() || g.kind<PF_GEOM_BOX || g.kind>PF_GEOM_HEIGHTFIELD
                || !pf_pose_valid(g.local) || !pf_vec_valid(g.size) || !pf_material_valid(g.material))
            return "invalid geometry or material";
        if ((g.kind==PF_GEOM_PLANE || g.kind==PF_GEOM_HEIGHTFIELD) && g.link>=0 && root[g.link]>=0)
            return "plane and heightfield must be static";
        if (g.kind<=PF_GEOM_CAPSULE && (g.size.x<=0
                || (g.kind==PF_GEOM_BOX && (g.size.y<=0 || g.size.z<=0))
                || ((g.kind==PF_GEOM_CYLINDER || g.kind==PF_GEOM_CAPSULE) && g.size.y<0)))
            return "invalid primitive dimensions";
        if (g.kind==PF_GEOM_CONVEX && (g.asset<0 || g.asset>=(int)h.convexes.size())) return "convex asset missing";
        if (g.kind==PF_GEOM_HEIGHTFIELD && (g.asset<0 || g.asset>=(int)h.heightfields.size())) return "heightfield asset missing";
        if (g.kind<=PF_GEOM_CAPSULE && g.material.margin!=0)
            return "primitive-pair speculative margins are not implemented";
    }
    for (const PfPair& p:h.excluded_links)
        if (p.a < -1 || p.b < -1 || p.a>=(int)h.links.size() || p.b>=(int)h.links.size()) return "exclusion index out of range";
    if (h.explicit_pairs) {
        // Explicit pairs override mask, parent/child and exclusion filtering.
        // Still reject pairs with no relative mobility and duplicate entries.
        for (size_t i=0;i<h.pairs.size();++i) {
            PfPair p=h.pairs[i];
            if (p.a<0 || p.b<0 || p.a>=(int)h.geoms.size() || p.b>=(int)h.geoms.size() || p.a==p.b)
                return "explicit geometry pair index out of range";
            const PfGeom& a=h.geoms[p.a]; const PfGeom& b=h.geoms[p.b];
            int la=a.link,lb=b.link;
            if (la==lb || ((la<0 || root[la]<0) && (lb<0 || root[lb]<0))
                    || (la>=0 && lb>=0 && welded[la]==welded[lb]))
                return "explicit pair has no relative mobility";
            if (!pf_geom_pair_supported(a.kind,b.kind)) return "unsupported explicit collision pair";
            for (size_t j=0;j<i;++j) {
                PfPair previous=h.pairs[j];
                if ((p.a==previous.a && p.b==previous.b) || (p.a==previous.b && p.b==previous.a))
                    return "duplicate explicit collision pair";
            }
        }
    } else {
        h.pairs.clear();
        for (int a=0;a<(int)h.geoms.size();++a) for (int b=a+1;b<(int)h.geoms.size();++b) {
            const PfGeom& ga=h.geoms[a]; const PfGeom& gb=h.geoms[b];
            int la=ga.link,lb=gb.link;
            if (!(ga.type&gb.affinity) && !(gb.type&ga.affinity)) continue;
            if (la==lb || ((la<0 || root[la]<0) && (lb<0 || root[lb]<0))) continue;
            if (la>=0 && lb>=0 && welded[la]==welded[lb]) continue;
            if (h.exclude_parent_child && la>=0 && lb>=0
                    && (h.links[la].parent==lb || h.links[lb].parent==la)) continue;
            bool excluded=false;
            for (PfPair p:h.excluded_links) excluded |= (p.a==la && p.b==lb) || (p.a==lb && p.b==la);
            if (excluded) continue;
            if (!pf_geom_pair_supported(ga.kind,gb.kind)) return "unsupported enabled collision pair";
            h.pairs.push_back({a,b});
        }
    }
    for (const PfActuator& a:h.actuators) {
        if (a.link<0 || a.link>=(int)h.links.size() || a.kind<PF_MOTOR || a.kind>PF_VELOCITY_SERVO)
            return "invalid actuator";
        int kind=h.links[a.link].joint;
        if (kind!=PF_HINGE && kind!=PF_SLIDE) return "actuator requires scalar joint";
        if (!pf_number(a.gear) || a.gear==0 || !pf_nonnegative(a.kp) || !pf_nonnegative(a.kd)
                || !pf_nonnegative(a.time_constant) || !pf_number(a.control_min) || !pf_number(a.control_max)
                || a.control_min>a.control_max || !pf_number(a.force_min) || !pf_number(a.force_max)
                || a.force_min>a.force_max) return "invalid actuator parameters";
    }
    for (const PfEquality& e:h.equalities) {
        if (e.kind<PF_CONNECT || e.kind>PF_COUPLE || e.a<0 || e.a>=(int)h.links.size()
                || e.b < -1 || e.b>=(int)h.links.size() || !pf_vec_valid(e.anchor_a)
                || !pf_vec_valid(e.anchor_b) || !pf_number(e.time_constant) || e.time_constant<=0
                || !pf_nonnegative(e.damping_ratio)) return "invalid equality";
        if (e.kind==PF_WELD && !pf_pose_valid({pf_v3(0,0,0),e.relative_rotation})) return "invalid weld rotation";
        if (e.a==e.b || (root[e.a]<0 && (e.b<0 || root[e.b]<0)))
            return "equality has no independent movable bodies";
        if (e.kind==PF_COUPLE && (e.b<0 || !pf_number(e.ratio) || !pf_number(e.offset)
                || pf_joint_nq(h.links[e.a].joint)!=1 || pf_joint_nq(h.links[e.b].joint)!=1)) return "invalid scalar coupling";
        if (e.kind==PF_COUPLE && h.links[e.a].dof==h.links[e.b].dof)
            return "scalar equality must connect independent coordinates";
    }
    for (const PfSite& site:h.sites)
        if (site.link < -1 || site.link>=(int)h.links.size() || !pf_pose_valid(site.local)) return "invalid site";
    m.links=h.links.data(); m.link_count=(int)h.links.size(); m.nq=nq; m.nv=nv;
    m.dofs=h.dofs.data(); m.initial_qpos=h.initial_qpos.data();
    m.geoms=h.geoms.data(); m.geom_count=(int)h.geoms.size();
    m.pairs=h.pairs.data(); m.pair_count=(int)h.pairs.size();
    m.actuators=h.actuators.data(); m.actuator_count=(int)h.actuators.size();
    m.equalities=h.equalities.data(); m.equality_count=(int)h.equalities.size();
    m.sites=h.sites.data(); m.site_count=(int)h.sites.size();
    m.convexes=h.convexes.data(); m.convex_count=(int)h.convexes.size();
    m.vertices=h.vertices.data(); m.vertex_count=(int)h.vertices.size();
    m.faces=h.faces.data(); m.face_count=(int)h.faces.size();
    m.heightfields=h.heightfields.data(); m.heightfield_count=(int)h.heightfields.size();
    m.heights=h.heights.data(); m.height_count=(int)h.heights.size();
    pf_compile_topology(h.links,nv,h.topology,m);
    return nullptr;
}
