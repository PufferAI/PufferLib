#pragma once
#include <mujoco/mujoco.h>
#include "model_builder.cuh"
#include "state.cuh"

// Offline only. This imports rigid model data, NOT MuJoCo's contact equations.
// Explicit opt-in is required because the native constraint model differs.
// The runtime neither includes nor links against MuJoCo.
static inline PfVec3 pf_mj_vec(const mjtNum* p) { return {(float)p[0],(float)p[1],(float)p[2]}; }
static inline PfQuat pf_mj_quat(const mjtNum* p) { return {(float)p[0],(float)p[1],(float)p[2],(float)p[3]}; }
static inline const char* pf_import_mujoco(const mjModel* source,bool native_constraints,
        PfModelStorage& out,PfModel& model,PfStepOptions& options) {
    if (!source) return "null MuJoCo model";
    if (!native_constraints) return "explicit native-constraint opt-in required; this is not MuJoCo trajectory parity";
    const mjModel& s=*source;
    if (s.nbody>INT_MAX || s.nv>INT_MAX || s.nq>INT_MAX || s.ngeom>INT_MAX)
        return "MuJoCo model exceeds native index range";
    if (s.ntendon || s.nflex || s.nplugin || s.neq || s.npair || s.nsensor || s.nmocap)
        return "import currently excludes tendons, flex, plugins, equalities, explicit pairs, sensor declarations, and mocap";
    if (s.opt.density!=0 || s.opt.viscosity!=0 || s.opt.enableflags!=0
            || (s.opt.disableflags & ~mjDSBL_FILTERPARENT)!=0 || s.opt.disableactuator!=0)
        return "unsupported fluid forces or option flags";
#if mjVERSION_HEADER >= 3014000
    for (int j=0;j<s.njnt;++j) for (int k=0;k<mjNPOLY;++k)
        if (s.jnt_stiffnesspoly[j*mjNPOLY+k]!=0) return "nonlinear joint springs not imported";
    for (int d=0;d<s.nv;++d) for (int k=0;k<mjNPOLY;++k)
        if (s.dof_dampingpoly[d*mjNPOLY+k]!=0) return "nonlinear joint damping not imported";
#endif
    PfModelStorage h;
    h.exclude_parent_child=!(s.opt.disableflags & mjDSBL_FILTERPARENT);
    h.links.resize((size_t)s.nbody-1); h.dofs.resize(s.nv); h.initial_qpos.resize(s.nq);
    for (int i=0;i<s.nq;++i) h.initial_qpos[i]=(float)s.qpos0[i];
    for (int b=1;b<s.nbody;++b) {
        if (s.body_jntnum[b]>1 || s.body_gravcomp[b]!=0) return "import requires at most one joint per body and no gravcomp";
        PfLink& l=h.links[b-1]; l.parent=s.body_parentid[b]-1; l.joint=PF_FIXED;
        l.rest={pf_mj_vec(s.body_pos+3*b),pf_mj_quat(s.body_quat+4*b)};
        l.center=pf_mj_vec(s.body_ipos+3*b); l.inertia_rotation=pf_mj_quat(s.body_iquat+4*b);
        l.mass=(float)s.body_mass[b]; l.inertia=pf_mj_vec(s.body_inertia+3*b);
        if (!s.body_jntnum[b]) continue;
        int j=s.body_jntadr[b],type=s.jnt_type[j],qa=s.jnt_qposadr[j],va=s.jnt_dofadr[j];
        l.joint=type==mjJNT_FREE?PF_FREE:type==mjJNT_BALL?PF_BALL:type==mjJNT_SLIDE?PF_SLIDE:PF_HINGE;
        l.anchor=pf_mj_vec(s.jnt_pos+3*j); l.axis=pf_mj_vec(s.jnt_axis+3*j);
        if ((l.joint==PF_HINGE || l.joint==PF_SLIDE) && s.qpos0[qa]!=0)
            return "nonzero scalar joint reference requires coordinate remapping before import";
        l.limited=s.jnt_limited[j]; l.lower=(float)s.jnt_range[2*j]; l.upper=(float)s.jnt_range[2*j+1];
        if (s.jnt_actfrclimited[j] || s.jnt_margin[j]!=0) return "joint actuator-force limits and limit margins not imported";
        for (int k=0;k<pf_joint_nv(l.joint);++k) {
            PfDof& d=h.dofs[va+k];
            d.armature=(float)s.dof_armature[va+k]; d.damping=(float)s.dof_damping[va+k];
            d.friction_loss=(float)s.dof_frictionloss[va+k];
            d.stiffness=(float)s.jnt_stiffness[j]; d.spring_reference=(float)s.qpos_spring[qa];
        }
    }
    for (int i=0;i<s.ngeom;++i) {
        int type=s.geom_type[i];
        if (type!=mjGEOM_BOX && type!=mjGEOM_SPHERE && type!=mjGEOM_CAPSULE
                && type!=mjGEOM_CYLINDER && type!=mjGEOM_PLANE) return "import supports primitive geometry; use native assets for convex/heightfield";
        if (s.geom_margin[i]!=0 || s.geom_gap[i]!=0 || s.geom_priority[i]!=0)
            return "geom margins, gaps, and priorities not imported";
        PfGeom g={}; g.link=s.geom_bodyid[i]-1;
        g.kind=type==mjGEOM_BOX?PF_GEOM_BOX:type==mjGEOM_SPHERE?PF_GEOM_SPHERE:
            type==mjGEOM_CAPSULE?PF_GEOM_CAPSULE:type==mjGEOM_CYLINDER?PF_GEOM_CYLINDER:PF_GEOM_PLANE;
        g.local={pf_mj_vec(s.geom_pos+3*i),pf_mj_quat(s.geom_quat+4*i)};
        g.size=pf_mj_vec(s.geom_size+3*i);
        if (g.kind==PF_GEOM_CAPSULE || g.kind==PF_GEOM_CYLINDER || g.kind==PF_GEOM_PLANE)
            g.local.rotation=pf_quat_multiply(g.local.rotation,pf_quat_from_axis_angle(pf_v3(1,0,0),1.5707963267948966f));
        g.type=s.geom_contype[i]; g.affinity=s.geom_conaffinity[i];
        g.material={(float)s.geom_friction[3*i],(float)s.geom_friction[3*i+1],(float)s.geom_friction[3*i+2],
            (float)s.geom_solref[2*i],(float)s.geom_solref[2*i+1],(float)s.geom_solimp[mjNIMP*i],0,s.geom_condim[i]};
        h.geoms.push_back(g);
    }
    for (int i=0;i<s.nexclude;++i) {
        unsigned int signature=(unsigned int)s.exclude_signature[i];
        h.excluded_links.push_back({(int)(signature>>16)-1,(int)(signature&65535)-1});
    }
    // SDK 3.14 introduces multiple inputs/outputs per actuator. Accept only
    // scalar one-to-one ordering; older SDKs use nu for the actuator count.
#if mjVERSION_HEADER >= 3014000
    int count=(int)s.nactuator;
    if (s.nu!=count || s.nout!=count) return "multi-input/output actuators not imported";
#else
    int count=(int)s.nu;
#endif
    for (int i=0;i<count;++i) {
#if mjVERSION_HEADER >= 3014000
        if (s.actuator_ctrladr[i]!=i || s.actuator_outadr[i]!=i
                || s.actuator_damping[i]!=0 || s.actuator_armature[i]!=0
                || s.actuator_delay[i]!=0) return "unsupported actuator ordering, passive parameters, or delay";
        for (int k=0;k<mjNPOLY;++k)
            if (s.actuator_dampingpoly[i*mjNPOLY+k]!=0) return "nonlinear actuator damping not imported";
#endif
        if (s.actuator_trntype[i]!=mjTRN_JOINT || s.actuator_gaintype[i]!=mjGAIN_FIXED
                || s.actuator_actlimited[i] || s.actuator_dyntype[i]!=mjDYN_NONE)
            return "import requires stateless fixed-gain scalar joint actuators";
        for (int k=1;k<6;++k) if (s.actuator_gear[6*i+k]!=0) return "non-scalar actuator gear";
        int joint=s.actuator_trnid[2*i];
        if (joint<0 || joint>=s.njnt) return "invalid actuator joint";
        PfActuator a={}; a.link=s.jnt_bodyid[joint]-1; a.gear=(float)s.actuator_gear[6*i];
        a.control_min=s.actuator_ctrllimited[i]?(float)s.actuator_ctrlrange[2*i]:-1.0e20f;
        a.control_max=s.actuator_ctrllimited[i]?(float)s.actuator_ctrlrange[2*i+1]:1.0e20f;
        a.force_min=s.actuator_forcelimited[i]?(float)s.actuator_forcerange[2*i]:-1.0e20f;
        a.force_max=s.actuator_forcelimited[i]?(float)s.actuator_forcerange[2*i+1]:1.0e20f;
        float gain=(float)s.actuator_gainprm[mjNGAIN*i];
        const mjtNum* bias=s.actuator_biasprm+mjNBIAS*i;
        if (s.actuator_biastype[i]==mjBIAS_NONE) {
            // Native motor control is actuator force. Preserve arbitrary gain
            // separately instead of changing transmission/force-limit units.
            if (gain!=1) return "motor gain must be one; rescale controls explicitly";
            a.kind=PF_MOTOR;
        } else if (s.actuator_biastype[i]==mjBIAS_AFFINE && bias[0]==0
                && bias[1]<=0 && bias[2]<=0) {
            if (bias[1]<0 && gain==(float)-bias[1]) {
                a.kind=PF_POSITION_SERVO; a.kp=gain; a.kd=(float)-bias[2];
            } else if (bias[1]==0 && gain==(float)-bias[2]) {
                a.kind=PF_VELOCITY_SERVO; a.kd=gain;
            } else return "unsupported actuator gain/bias combination";
        } else return "unsupported actuator bias";
        h.actuators.push_back(a);
    }
    for (int i=0;i<s.nsite;++i)
        h.sites.push_back({s.site_bodyid[i]-1,{pf_mj_vec(s.site_pos+3*i),pf_mj_quat(s.site_quat+4*i)}});
    PfModel view; const char* error=pf_compile_model(h,view);
    if (error) return error;
    if (view.nq!=s.nq || view.nv!=s.nv) return "compiled coordinate layout mismatch";
    options=pf_default_step_options(); options.dt=(float)s.opt.timestep; options.gravity=pf_mj_vec(s.opt.gravity);
    out=std::move(h); return pf_compile_model(out,model);
}
