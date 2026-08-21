// MuJoCo-style rigid body physics for PufferLib envs. CUDA C99, fixed capacity,
// float. Ports mj_step's pipeline (same algorithms, names and layouts as the
// MuJoCo engine) for the features the classic Gym models use: free/ball/hinge/
// slide joints, plane/sphere/capsule geoms, motor actuators, joint springs and
// dampers, joint limits and pyramidal friction contacts with MuJoCo's soft
// constraint model, semi-implicit Euler with implicit damping, and RK4.
// Models are compiled from MJCF by ocean/mujoco/mjcf2bin.py and loaded with
// mj_loadModel; arrays have fixed MJ_MAX_* capacity (defaults below, envs
// define tighter ones for their model before including) and runtime counts.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __CUDACC__
#define MJ_HD __host__ __device__
#else
#define MJ_HD
#endif

#define MJ_MINVAL 1e-15f
#define MJ_MINIMP 0.0001f
#define MJ_MAXIMP 0.9999f
#ifndef MJ_MAX_NQ
#define MJ_MAX_NQ 32
#endif
#ifndef MJ_MAX_NV
#define MJ_MAX_NV 32
#endif
#ifndef MJ_MAX_NBODY
#define MJ_MAX_NBODY 16
#endif
#ifndef MJ_MAX_NJNT
#define MJ_MAX_NJNT 24
#endif
#ifndef MJ_MAX_NGEOM
#define MJ_MAX_NGEOM 24
#endif
#ifndef MJ_MAX_NSITE
#define MJ_MAX_NSITE 4
#endif
#ifndef MJ_MAX_NU
#define MJ_MAX_NU 24
#endif
#ifndef MJ_MAXCON
#define MJ_MAXCON 48
#endif
// constraint rows: 1 per active limit, 1 per frictionless contact, 2*(condim-1) otherwise
#ifndef MJ_MAXEFC
#define MJ_MAXEFC 128
#endif
#define MJ_MAGIC 0x4e424a4d
enum {MJ_JNT_FREE, MJ_JNT_BALL, MJ_JNT_SLIDE, MJ_JNT_HINGE};
enum {MJ_GEOM_PLANE, MJ_GEOM_HFIELD, MJ_GEOM_SPHERE, MJ_GEOM_CAPSULE, MJ_GEOM_ELLIPSOID,
    MJ_GEOM_CYLINDER, MJ_GEOM_BOX};
enum {MJ_INT_EULER, MJ_INT_RK4};

typedef struct {
    int nq, nv, nbody, njnt, ngeom, nsite, nu;
    float opt_timestep;
    float opt_gravity[3];
    int opt_integrator;
    float opt_impratio;
    int body_parentid[MJ_MAX_NBODY];
    int body_rootid[MJ_MAX_NBODY];
    int body_weldid[MJ_MAX_NBODY];
    int body_jntadr[MJ_MAX_NBODY];
    int body_jntnum[MJ_MAX_NBODY];
    int body_dofadr[MJ_MAX_NBODY];
    int body_dofnum[MJ_MAX_NBODY];
    float body_pos[MJ_MAX_NBODY][3];
    float body_quat[MJ_MAX_NBODY][4];
    float body_ipos[MJ_MAX_NBODY][3];
    float body_iquat[MJ_MAX_NBODY][4];
    float body_mass[MJ_MAX_NBODY];
    float body_subtreemass[MJ_MAX_NBODY];
    float body_inertia[MJ_MAX_NBODY][3];
    float body_invweight0[MJ_MAX_NBODY][2];
    int jnt_type[MJ_MAX_NJNT];
    int jnt_qposadr[MJ_MAX_NJNT];
    int jnt_dofadr[MJ_MAX_NJNT];
    int jnt_limited[MJ_MAX_NJNT];
    float jnt_axis[MJ_MAX_NJNT][3];
    float jnt_pos[MJ_MAX_NJNT][3];
    float jnt_range[MJ_MAX_NJNT][2];
    float jnt_margin[MJ_MAX_NJNT];
    float jnt_stiffness[MJ_MAX_NJNT];
    float jnt_solref[MJ_MAX_NJNT][2];
    float jnt_solimp[MJ_MAX_NJNT][5];
    int dof_bodyid[MJ_MAX_NV];
    int dof_jntid[MJ_MAX_NV];
    int dof_parentid[MJ_MAX_NV];
    float dof_armature[MJ_MAX_NV];
    float dof_damping[MJ_MAX_NV];
    float dof_invweight0[MJ_MAX_NV];
    int geom_type[MJ_MAX_NGEOM];
    int geom_bodyid[MJ_MAX_NGEOM];
    int geom_contype[MJ_MAX_NGEOM];
    int geom_conaffinity[MJ_MAX_NGEOM];
    int geom_condim[MJ_MAX_NGEOM];
    int geom_priority[MJ_MAX_NGEOM];
    float geom_solmix[MJ_MAX_NGEOM];
    float geom_size[MJ_MAX_NGEOM][3];
    float geom_pos[MJ_MAX_NGEOM][3];
    float geom_quat[MJ_MAX_NGEOM][4];
    float geom_friction[MJ_MAX_NGEOM][3];
    float geom_solref[MJ_MAX_NGEOM][2];
    float geom_solimp[MJ_MAX_NGEOM][5];
    float geom_margin[MJ_MAX_NGEOM];
    float geom_gap[MJ_MAX_NGEOM];
    float geom_rbound[MJ_MAX_NGEOM];
    int site_bodyid[MJ_MAX_NSITE];
    float site_pos[MJ_MAX_NSITE][3];
    float site_quat[MJ_MAX_NSITE][4];
    int actuator_trnid[MJ_MAX_NU];
    float actuator_gear[MJ_MAX_NU];
    int actuator_ctrllimited[MJ_MAX_NU];
    float actuator_ctrlrange[MJ_MAX_NU][2];
    float qpos0[MJ_MAX_NQ];
    float qpos_spring[MJ_MAX_NQ];
} MjModel;

typedef struct {
    int geom1, geom2, dim;
    float dist;
    float pos[3];
    float frame[9];
    float includemargin;
    float friction[5];
    float solref[2];
    float solimp[5];
    int efc_address;
} MjContact;

// Constraint solver scratch: MJ_SCRATCH floats per env
#define MJ_SCRATCH (2*MJ_MAXEFC*MJ_MAXEFC + MJ_MAXEFC*MJ_MAX_NV)

typedef struct {
    float* efc_AR;
    float* efc_ARfree;
    float* efc_MinvJT;
    int efc_stride;
    float time;
    float qpos[MJ_MAX_NQ];
    float qvel[MJ_MAX_NV];
    float ctrl[MJ_MAX_NU];
    float qacc[MJ_MAX_NV];
    float xpos[MJ_MAX_NBODY][3];
    float xipos[MJ_MAX_NBODY][3];
    float xquat[MJ_MAX_NBODY][4];
    float xmat[MJ_MAX_NBODY][9];
    float ximat[MJ_MAX_NBODY][9];
    float xanchor[MJ_MAX_NJNT][3];
    float xaxis[MJ_MAX_NJNT][3];
    float geom_xpos[MJ_MAX_NGEOM][3];
    float geom_xmat[MJ_MAX_NGEOM][9];
    float site_xpos[MJ_MAX_NSITE][3];
    float site_xmat[MJ_MAX_NSITE][9];
    float subtree_com[MJ_MAX_NBODY][3];
    float cinert[MJ_MAX_NBODY][10];
    float cdof[MJ_MAX_NV][6];
    float cvel[MJ_MAX_NBODY][6];
    float cdof_dot[MJ_MAX_NV][6];
    float qM[MJ_MAX_NV][MJ_MAX_NV];
    float qLD[MJ_MAX_NV][MJ_MAX_NV];
    float qfrc_bias[MJ_MAX_NV];
    float qfrc_passive[MJ_MAX_NV];
    float qfrc_actuator[MJ_MAX_NV];
    float qfrc_smooth[MJ_MAX_NV];
    float qacc_smooth[MJ_MAX_NV];
    float qfrc_constraint[MJ_MAX_NV];
    float cacc[MJ_MAX_NBODY][6];
    float cfrc_int[MJ_MAX_NBODY][6];
    float cfrc_ext[MJ_MAX_NBODY][6];
    int ncon;
    MjContact contact[MJ_MAXCON];
    int nefc;
    float efc_J[MJ_MAXEFC][MJ_MAX_NV];
    float efc_pos[MJ_MAXEFC];
    float efc_margin[MJ_MAXEFC];
    float efc_R[MJ_MAXEFC];
    float efc_aref[MJ_MAXEFC];
    float efc_force[MJ_MAXEFC];
} MjData;

MJ_HD void mj_makeData(MjData* d, float* scratch, int stride) {
    d->efc_AR = scratch;
    d->efc_ARfree = scratch + MJ_MAXEFC*MJ_MAXEFC*stride;
    d->efc_MinvJT = scratch + 2*MJ_MAXEFC*MJ_MAXEFC*stride;
    d->efc_stride = stride;
}

void mj_read(FILE* fp, void* dst, int count, int size) {
    assert((int)fread(dst, size, count, fp) == count && "truncated model file");
}

// Load a model compiled by ocean/mujoco/mjcf2bin.py (same field order)
void mj_loadModel(MjModel* m, const char* path) {
    FILE* fp = fopen(path, "rb");
    assert(fp && "cannot open model file");
    int head[9];
    mj_read(fp, head, 9, sizeof(int));
    assert(head[0] == MJ_MAGIC && head[1] == 1 && "bad model file");
    m->nq = head[2];
    m->nv = head[3];
    m->nbody = head[4];
    m->njnt = head[5];
    m->ngeom = head[6];
    m->nsite = head[7];
    m->nu = head[8];
    assert(m->nq <= MJ_MAX_NQ && m->nv <= MJ_MAX_NV && m->nbody <= MJ_MAX_NBODY
        && m->njnt <= MJ_MAX_NJNT && m->ngeom <= MJ_MAX_NGEOM && m->nsite <= MJ_MAX_NSITE
        && m->nu <= MJ_MAX_NU && "model exceeds MJ_MAX_* capacity");
    mj_read(fp, &m->opt_timestep, 1, sizeof(float));
    mj_read(fp, m->opt_gravity, 3, sizeof(float));
    mj_read(fp, &m->opt_integrator, 1, sizeof(int));
    mj_read(fp, &m->opt_impratio, 1, sizeof(float));
    mj_read(fp, m->body_parentid, m->nbody, sizeof(int));
    mj_read(fp, m->body_rootid, m->nbody, sizeof(int));
    mj_read(fp, m->body_weldid, m->nbody, sizeof(int));
    mj_read(fp, m->body_jntadr, m->nbody, sizeof(int));
    mj_read(fp, m->body_jntnum, m->nbody, sizeof(int));
    mj_read(fp, m->body_dofadr, m->nbody, sizeof(int));
    mj_read(fp, m->body_dofnum, m->nbody, sizeof(int));
    mj_read(fp, m->body_pos, 3*m->nbody, sizeof(float));
    mj_read(fp, m->body_quat, 4*m->nbody, sizeof(float));
    mj_read(fp, m->body_ipos, 3*m->nbody, sizeof(float));
    mj_read(fp, m->body_iquat, 4*m->nbody, sizeof(float));
    mj_read(fp, m->body_mass, m->nbody, sizeof(float));
    mj_read(fp, m->body_subtreemass, m->nbody, sizeof(float));
    mj_read(fp, m->body_inertia, 3*m->nbody, sizeof(float));
    mj_read(fp, m->body_invweight0, 2*m->nbody, sizeof(float));
    mj_read(fp, m->jnt_type, m->njnt, sizeof(int));
    mj_read(fp, m->jnt_qposadr, m->njnt, sizeof(int));
    mj_read(fp, m->jnt_dofadr, m->njnt, sizeof(int));
    mj_read(fp, m->jnt_limited, m->njnt, sizeof(int));
    mj_read(fp, m->jnt_axis, 3*m->njnt, sizeof(float));
    mj_read(fp, m->jnt_pos, 3*m->njnt, sizeof(float));
    mj_read(fp, m->jnt_range, 2*m->njnt, sizeof(float));
    mj_read(fp, m->jnt_margin, m->njnt, sizeof(float));
    mj_read(fp, m->jnt_stiffness, m->njnt, sizeof(float));
    mj_read(fp, m->jnt_solref, 2*m->njnt, sizeof(float));
    mj_read(fp, m->jnt_solimp, 5*m->njnt, sizeof(float));
    mj_read(fp, m->dof_bodyid, m->nv, sizeof(int));
    mj_read(fp, m->dof_jntid, m->nv, sizeof(int));
    mj_read(fp, m->dof_parentid, m->nv, sizeof(int));
    mj_read(fp, m->dof_armature, m->nv, sizeof(float));
    mj_read(fp, m->dof_damping, m->nv, sizeof(float));
    mj_read(fp, m->dof_invweight0, m->nv, sizeof(float));
    mj_read(fp, m->geom_type, m->ngeom, sizeof(int));
    mj_read(fp, m->geom_bodyid, m->ngeom, sizeof(int));
    mj_read(fp, m->geom_contype, m->ngeom, sizeof(int));
    mj_read(fp, m->geom_conaffinity, m->ngeom, sizeof(int));
    mj_read(fp, m->geom_condim, m->ngeom, sizeof(int));
    mj_read(fp, m->geom_priority, m->ngeom, sizeof(int));
    mj_read(fp, m->geom_solmix, m->ngeom, sizeof(float));
    mj_read(fp, m->geom_size, 3*m->ngeom, sizeof(float));
    mj_read(fp, m->geom_pos, 3*m->ngeom, sizeof(float));
    mj_read(fp, m->geom_quat, 4*m->ngeom, sizeof(float));
    mj_read(fp, m->geom_friction, 3*m->ngeom, sizeof(float));
    mj_read(fp, m->geom_solref, 2*m->ngeom, sizeof(float));
    mj_read(fp, m->geom_solimp, 5*m->ngeom, sizeof(float));
    mj_read(fp, m->geom_margin, m->ngeom, sizeof(float));
    mj_read(fp, m->geom_gap, m->ngeom, sizeof(float));
    mj_read(fp, m->geom_rbound, m->ngeom, sizeof(float));
    mj_read(fp, m->site_bodyid, m->nsite, sizeof(int));
    mj_read(fp, m->site_pos, 3*m->nsite, sizeof(float));
    mj_read(fp, m->site_quat, 4*m->nsite, sizeof(float));
    mj_read(fp, m->actuator_trnid, m->nu, sizeof(int));
    mj_read(fp, m->actuator_gear, m->nu, sizeof(float));
    mj_read(fp, m->actuator_ctrllimited, m->nu, sizeof(int));
    mj_read(fp, m->actuator_ctrlrange, 2*m->nu, sizeof(float));
    mj_read(fp, m->qpos0, m->nq, sizeof(float));
    mj_read(fp, m->qpos_spring, m->nq, sizeof(float));
    fclose(fp);
}

// Vector, quaternion and spatial algebra (mju_*)

MJ_HD float mju_dot3(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

MJ_HD void mju_cross(float* res, const float* a, const float* b) {
    float t0 = a[1]*b[2] - a[2]*b[1];
    float t1 = a[2]*b[0] - a[0]*b[2];
    float t2 = a[0]*b[1] - a[1]*b[0];
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
}

MJ_HD float mju_normalize3(float* v) {
    float norm = sqrtf(mju_dot3(v, v));
    if (norm < MJ_MINVAL) {
        v[0] = 1.0f;
        v[1] = 0.0f;
        v[2] = 0.0f;
    } else {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
    return norm;
}

MJ_HD void mju_normalize4(float* q) {
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm < MJ_MINVAL) {
        q[0] = 1.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 0.0f;
    } else {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
}

MJ_HD void mju_mulQuat(float* res, const float* qa, const float* qb) {
    res[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3];
    res[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2];
    res[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1];
    res[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0];
}

MJ_HD void mju_rotVecQuat(float* res, const float* vec, const float* quat) {
    float t0 = quat[0]*vec[0] + quat[2]*vec[2] - quat[3]*vec[1];
    float t1 = quat[0]*vec[1] + quat[3]*vec[0] - quat[1]*vec[2];
    float t2 = quat[0]*vec[2] + quat[1]*vec[1] - quat[2]*vec[0];
    res[0] = vec[0] + 2.0f*(quat[2]*t2 - quat[3]*t1);
    res[1] = vec[1] + 2.0f*(quat[3]*t0 - quat[1]*t2);
    res[2] = vec[2] + 2.0f*(quat[1]*t1 - quat[2]*t0);
}

MJ_HD void mju_quat2Mat(float* res, const float* q) {
    float q00 = q[0]*q[0], q01 = q[0]*q[1], q02 = q[0]*q[2], q03 = q[0]*q[3];
    float q11 = q[1]*q[1], q12 = q[1]*q[2], q13 = q[1]*q[3];
    float q22 = q[2]*q[2], q23 = q[2]*q[3], q33 = q[3]*q[3];
    res[0] = q00 + q11 - q22 - q33;
    res[4] = q00 - q11 + q22 - q33;
    res[8] = q00 - q11 - q22 + q33;
    res[1] = 2.0f*(q12 - q03);
    res[2] = 2.0f*(q13 + q02);
    res[3] = 2.0f*(q12 + q03);
    res[5] = 2.0f*(q23 - q01);
    res[6] = 2.0f*(q13 - q02);
    res[7] = 2.0f*(q23 + q01);
}

MJ_HD void mju_axisAngle2Quat(float* res, const float* axis, float angle) {
    float s = sinf(0.5f*angle);
    res[0] = cosf(0.5f*angle);
    res[1] = axis[0]*s;
    res[2] = axis[1]*s;
    res[3] = axis[2]*s;
}

// Integrate quaternion by angular velocity expressed in the local frame
MJ_HD void mju_quatIntegrate(float* quat, const float* vel, float scale) {
    float axis[3] = {vel[0], vel[1], vel[2]};
    float angle = scale*mju_normalize3(axis);
    float qrot[4];
    mju_axisAngle2Quat(qrot, axis, angle);
    mju_normalize4(quat);
    mju_mulQuat(quat, quat, qrot);
}

MJ_HD void mju_mulMatTVec3(float* res, const float* mat, const float* vec) {
    res[0] = mat[0]*vec[0] + mat[3]*vec[1] + mat[6]*vec[2];
    res[1] = mat[1]*vec[0] + mat[4]*vec[1] + mat[7]*vec[2];
    res[2] = mat[2]*vec[0] + mat[5]*vec[1] + mat[8]*vec[2];
}

MJ_HD void mju_mulMatVec3(float* res, const float* mat, const float* vec) {
    res[0] = mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2];
    res[1] = mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2];
    res[2] = mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2];
}

// Motion vectors are (angular, linear); cross products of motion and force
MJ_HD void mju_crossMotion(float* res, const float* vel, const float* v) {
    res[0] = -vel[2]*v[1] + vel[1]*v[2];
    res[1] = vel[2]*v[0] - vel[0]*v[2];
    res[2] = -vel[1]*v[0] + vel[0]*v[1];
    res[3] = -vel[2]*v[4] + vel[1]*v[5] - vel[5]*v[1] + vel[4]*v[2];
    res[4] = vel[2]*v[3] - vel[0]*v[5] + vel[5]*v[0] - vel[3]*v[2];
    res[5] = -vel[1]*v[3] + vel[0]*v[4] - vel[4]*v[0] + vel[3]*v[1];
}

MJ_HD void mju_crossForce(float* res, const float* vel, const float* f) {
    res[0] = -vel[2]*f[1] + vel[1]*f[2] - vel[5]*f[4] + vel[4]*f[5];
    res[1] = vel[2]*f[0] - vel[0]*f[2] + vel[5]*f[3] - vel[3]*f[5];
    res[2] = -vel[1]*f[0] + vel[0]*f[1] - vel[4]*f[3] + vel[3]*f[4];
    res[3] = -vel[2]*f[4] + vel[1]*f[5];
    res[4] = vel[2]*f[3] - vel[0]*f[5];
    res[5] = -vel[1]*f[3] + vel[0]*f[4];
}

// Spatial inertia (10: I_xx I_yy I_zz I_xy I_xz I_yz, m*com, m) in a frame
// displaced by dif from the body inertial frame, rotated by mat
MJ_HD void mju_inertCom(float* res, const float* inert, const float* mat, const float* dif,
    float mass) {
    float tmp[9] = {mat[0]*inert[0], mat[3]*inert[0], mat[6]*inert[0],
        mat[1]*inert[1], mat[4]*inert[1], mat[7]*inert[1],
        mat[2]*inert[2], mat[5]*inert[2], mat[8]*inert[2]};
    res[0] = mat[0]*tmp[0] + mat[1]*tmp[3] + mat[2]*tmp[6] + mass*(dif[1]*dif[1] + dif[2]*dif[2]);
    res[1] = mat[3]*tmp[1] + mat[4]*tmp[4] + mat[5]*tmp[7] + mass*(dif[0]*dif[0] + dif[2]*dif[2]);
    res[2] = mat[6]*tmp[2] + mat[7]*tmp[5] + mat[8]*tmp[8] + mass*(dif[0]*dif[0] + dif[1]*dif[1]);
    res[3] = mat[0]*tmp[1] + mat[1]*tmp[4] + mat[2]*tmp[7] - mass*dif[0]*dif[1];
    res[4] = mat[0]*tmp[2] + mat[1]*tmp[5] + mat[2]*tmp[8] - mass*dif[0]*dif[2];
    res[5] = mat[3]*tmp[2] + mat[4]*tmp[5] + mat[5]*tmp[8] - mass*dif[1]*dif[2];
    res[6] = mass*dif[0];
    res[7] = mass*dif[1];
    res[8] = mass*dif[2];
    res[9] = mass;
}

MJ_HD void mju_mulInertVec(float* res, const float* i, const float* v) {
    res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5];
    res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5];
    res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4];
    res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3];
    res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4];
    res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5];
}

MJ_HD float mju_dot6(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5];
}

// xorshift32 uniform in [0, 1) and standard normal, for env resets
MJ_HD float mju_rand(unsigned int* rng) {
    unsigned int x = *rng ? *rng : 0x9e3779b9u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *rng = x;
    return (x >> 8)*(1.0f / 16777216.0f);
}

MJ_HD float mju_randn(unsigned int* rng) {
    float u1 = 1.0f - mju_rand(rng);
    float u2 = mju_rand(rng);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*(float)M_PI*u2);
}

// In-place Cholesky factor A = L L^T
MJ_HD void mju_cholFactor(float* A, int n, int lda, int es) {
    for (int j = 0; j < n; j++) {
        float d = A[(j*lda + j)*es];
        for (int k = 0; k < j; k++) {
            d -= A[(j*lda + k)*es]*A[(j*lda + k)*es];
        }
        d = sqrtf(d);
        A[(j*lda + j)*es] = d;
        for (int i = j + 1; i < n; i++) {
            float s = A[(i*lda + j)*es];
            for (int k = 0; k < j; k++) {
                s -= A[(i*lda + k)*es]*A[(j*lda + k)*es];
            }
            A[(i*lda + j)*es] = s / d;
        }
    }
}

MJ_HD void mju_cholSolve(const float* L, int n, int lda, int es, float* x) {
    for (int i = 0; i < n; i++) {
        float s = x[i];
        for (int k = 0; k < i; k++) {
            s -= L[(i*lda + k)*es]*x[k];
        }
        x[i] = s / L[(i*lda + i)*es];
    }
    for (int i = n - 1; i >= 0; i--) {
        float s = x[i];
        for (int k = i + 1; k < n; k++) {
            s -= L[(k*lda + i)*es]*x[k];
        }
        x[i] = s / L[(i*lda + i)*es];
    }
}

// Forward kinematics: body, inertial, geom and site frames, joint anchors/axes

MJ_HD void mj_local2Global(MjData* d, float* xpos, float* xmat, const float* pos,
    const float* quat, int body) {
    float tmp[3], q[4];
    mju_mulMatVec3(tmp, d->xmat[body], pos);
    xpos[0] = d->xpos[body][0] + tmp[0];
    xpos[1] = d->xpos[body][1] + tmp[1];
    xpos[2] = d->xpos[body][2] + tmp[2];
    mju_mulQuat(q, d->xquat[body], quat);
    mju_quat2Mat(xmat, q);
}

MJ_HD void mj_kinematics(const MjModel* m, MjData* d) {
    memset(d->xpos[0], 0, 3*sizeof(float));
    memset(d->xmat[0], 0, 9*sizeof(float));
    d->xquat[0][0] = 1.0f;
    d->xquat[0][1] = 0.0f;
    d->xquat[0][2] = 0.0f;
    d->xquat[0][3] = 0.0f;
    d->xmat[0][0] = 1.0f;
    d->xmat[0][4] = 1.0f;
    d->xmat[0][8] = 1.0f;
    for (int i = 1; i < m->nbody; i++) {
        float xpos[3], xquat[4];
        int jntadr = m->body_jntadr[i];
        int jntnum = m->body_jntnum[i];
        if (jntnum == 1 && m->jnt_type[jntadr] == MJ_JNT_FREE) {
            int qadr = m->jnt_qposadr[jntadr];
            memcpy(xpos, d->qpos + qadr, 3*sizeof(float));
            memcpy(xquat, d->qpos + qadr + 3, 4*sizeof(float));
            mju_normalize4(xquat);
            memcpy(d->xanchor[jntadr], xpos, 3*sizeof(float));
            memcpy(d->xaxis[jntadr], m->jnt_axis[jntadr], 3*sizeof(float));
        } else {
            int pid = m->body_parentid[i];
            mju_mulMatVec3(xpos, d->xmat[pid], m->body_pos[i]);
            xpos[0] += d->xpos[pid][0];
            xpos[1] += d->xpos[pid][1];
            xpos[2] += d->xpos[pid][2];
            mju_mulQuat(xquat, d->xquat[pid], m->body_quat[i]);
            for (int j = jntadr; j < jntadr + jntnum; j++) {
                int qadr = m->jnt_qposadr[j];
                int jtype = m->jnt_type[j];
                float xaxis[3], xanchor[3];
                mju_rotVecQuat(xaxis, m->jnt_axis[j], xquat);
                mju_rotVecQuat(xanchor, m->jnt_pos[j], xquat);
                xanchor[0] += xpos[0];
                xanchor[1] += xpos[1];
                xanchor[2] += xpos[2];
                if (jtype == MJ_JNT_SLIDE) {
                    float dq = d->qpos[qadr] - m->qpos0[qadr];
                    xpos[0] += xaxis[0]*dq;
                    xpos[1] += xaxis[1]*dq;
                    xpos[2] += xaxis[2]*dq;
                } else {
                    float qloc[4], vec[3];
                    if (jtype == MJ_JNT_BALL) {
                        memcpy(qloc, d->qpos + qadr, 4*sizeof(float));
                        mju_normalize4(qloc);
                    } else {
                        mju_axisAngle2Quat(qloc, m->jnt_axis[j], d->qpos[qadr] - m->qpos0[qadr]);
                    }
                    mju_mulQuat(xquat, xquat, qloc);
                    mju_rotVecQuat(vec, m->jnt_pos[j], xquat);
                    xpos[0] = xanchor[0] - vec[0];
                    xpos[1] = xanchor[1] - vec[1];
                    xpos[2] = xanchor[2] - vec[2];
                }
                memcpy(d->xanchor[j], xanchor, 3*sizeof(float));
                memcpy(d->xaxis[j], xaxis, 3*sizeof(float));
            }
        }
        mju_normalize4(xquat);
        memcpy(d->xpos[i], xpos, 3*sizeof(float));
        memcpy(d->xquat[i], xquat, 4*sizeof(float));
        mju_quat2Mat(d->xmat[i], xquat);
    }
    for (int i = 1; i < m->nbody; i++) {
        mj_local2Global(d, d->xipos[i], d->ximat[i], m->body_ipos[i], m->body_iquat[i], i);
    }
    for (int g = 0; g < m->ngeom; g++) {
        mj_local2Global(d, d->geom_xpos[g], d->geom_xmat[g], m->geom_pos[g], m->geom_quat[g],
            m->geom_bodyid[g]);
    }
    for (int s = 0; s < m->nsite; s++) {
        mj_local2Global(d, d->site_xpos[s], d->site_xmat[s], m->site_pos[s], m->site_quat[s],
            m->site_bodyid[s]);
    }
}

// Subtree centers of mass, spatial inertias and dof motion axes in the frame
// centered at the kinematic tree's subtree COM (world orientation)
MJ_HD void mj_comPos(const MjModel* m, MjData* d) {
    for (int i = 0; i < m->nbody; i++) {
        for (int k = 0; k < 3; k++) {
            d->subtree_com[i][k] = d->xipos[i][k]*m->body_mass[i];
        }
    }
    for (int i = m->nbody - 1; i > 0; i--) {
        int p = m->body_parentid[i];
        for (int k = 0; k < 3; k++) {
            d->subtree_com[p][k] += d->subtree_com[i][k];
        }
    }
    for (int i = 0; i < m->nbody; i++) {
        if (m->body_subtreemass[i] < MJ_MINVAL) {
            memcpy(d->subtree_com[i], d->xipos[i], 3*sizeof(float));
            continue;
        }
        for (int k = 0; k < 3; k++) {
            d->subtree_com[i][k] /= m->body_subtreemass[i];
        }
    }
    memset(d->cinert[0], 0, 10*sizeof(float));
    for (int i = 1; i < m->nbody; i++) {
        float* com = d->subtree_com[m->body_rootid[i]];
        float offset[3] = {d->xipos[i][0] - com[0], d->xipos[i][1] - com[1], d->xipos[i][2] - com[2]};
        mju_inertCom(d->cinert[i], m->body_inertia[i], d->ximat[i], offset, m->body_mass[i]);
    }
    for (int i = 1; i < m->nbody; i++) {
        float* com = d->subtree_com[m->body_rootid[i]];
        for (int j = m->body_jntadr[i]; j < m->body_jntadr[i] + m->body_jntnum[i]; j++) {
            int da = m->jnt_dofadr[j];
            int jtype = m->jnt_type[j];
            float offset[3] = {com[0] - d->xanchor[j][0], com[1] - d->xanchor[j][1],
                com[2] - d->xanchor[j][2]};
            if (jtype == MJ_JNT_SLIDE) {
                memset(d->cdof[da], 0, 6*sizeof(float));
                memcpy(d->cdof[da] + 3, d->xaxis[j], 3*sizeof(float));
            } else if (jtype == MJ_JNT_HINGE) {
                memcpy(d->cdof[da], d->xaxis[j], 3*sizeof(float));
                mju_cross(d->cdof[da] + 3, d->xaxis[j], offset);
            } else {
                int skip = 0;
                if (jtype == MJ_JNT_FREE) {
                    memset(d->cdof[da], 0, 18*sizeof(float));
                    d->cdof[da][3] = 1.0f;
                    d->cdof[da + 1][4] = 1.0f;
                    d->cdof[da + 2][5] = 1.0f;
                    skip = 3;
                }
                for (int k = 0; k < 3; k++) {
                    float axis[3] = {d->xmat[i][k], d->xmat[i][k + 3], d->xmat[i][k + 6]};
                    memcpy(d->cdof[da + skip + k], axis, 3*sizeof(float));
                    mju_cross(d->cdof[da + skip + k] + 3, axis, offset);
                }
            }
        }
    }
}

// Composite rigid body: dense mass matrix and its Cholesky factor
MJ_HD void mj_crb(const MjModel* m, MjData* d) {
    float crb[MJ_MAX_NBODY][10];
    memcpy(crb, d->cinert, sizeof(crb));
    for (int i = m->nbody - 1; i > 0; i--) {
        int p = m->body_parentid[i];
        if (p > 0) {
            for (int k = 0; k < 10; k++) {
                crb[p][k] += crb[i][k];
            }
        }
    }
    memset(d->qM, 0, sizeof(d->qM));
    for (int i = 0; i < m->nv; i++) {
        float buf[6];
        mju_mulInertVec(buf, crb[m->dof_bodyid[i]], d->cdof[i]);
        d->qM[i][i] = m->dof_armature[i];
        for (int j = i; j >= 0; j = m->dof_parentid[j]) {
            d->qM[i][j] += mju_dot6(d->cdof[j], buf);
            d->qM[j][i] = d->qM[i][j];
        }
    }
    memcpy(d->qLD, d->qM, sizeof(d->qM));
    mju_cholFactor(&d->qLD[0][0], m->nv, MJ_MAX_NV, 1);
}

// Body spatial velocities and dof axis derivatives (cvel x cdof)
MJ_HD void mj_comVel(const MjModel* m, MjData* d) {
    memset(d->cvel[0], 0, 6*sizeof(float));
    for (int i = 1; i < m->nbody; i++) {
        float cvel[6];
        memcpy(cvel, d->cvel[m->body_parentid[i]], sizeof(cvel));
        int bda = m->body_dofadr[i];
        int dofnum = m->body_dofnum[i];
        for (int j = 0; j < dofnum; j++) {
            int jtype = m->jnt_type[m->dof_jntid[bda + j]];
            int n = 1;
            if (jtype == MJ_JNT_FREE) {
                memset(d->cdof_dot[bda], 0, 18*sizeof(float));
                for (int k = 0; k < 3; k++) {
                    for (int c = 0; c < 6; c++) {
                        cvel[c] += d->cdof[bda + k][c]*d->qvel[bda + k];
                    }
                }
                j += 3;
                n = 3;
            } else if (jtype == MJ_JNT_BALL) {
                n = 3;
            }
            for (int k = 0; k < n; k++) {
                mju_crossMotion(d->cdof_dot[bda + j + k], cvel, d->cdof[bda + j + k]);
            }
            for (int k = 0; k < n; k++) {
                for (int c = 0; c < 6; c++) {
                    cvel[c] += d->cdof[bda + j + k][c]*d->qvel[bda + j + k];
                }
            }
            j += n - 1;
        }
        memcpy(d->cvel[i], cvel, sizeof(cvel));
    }
}

// Recursive Newton-Euler without acceleration
MJ_HD void mj_rne(const MjModel* m, MjData* d) {
    float cacc[MJ_MAX_NBODY][6], cfrc[MJ_MAX_NBODY][6];
    memset(cacc[0], 0, 6*sizeof(float));
    cacc[0][3] = -m->opt_gravity[0];
    cacc[0][4] = -m->opt_gravity[1];
    cacc[0][5] = -m->opt_gravity[2];
    for (int i = 1; i < m->nbody; i++) {
        int bda = m->body_dofadr[i];
        float tmp[6], tmp1[6];
        memcpy(cacc[i], cacc[m->body_parentid[i]], 6*sizeof(float));
        for (int j = bda; j < bda + m->body_dofnum[i]; j++) {
            for (int c = 0; c < 6; c++) {
                cacc[i][c] += d->cdof_dot[j][c]*d->qvel[j];
            }
        }
        mju_mulInertVec(cfrc[i], d->cinert[i], cacc[i]);
        mju_mulInertVec(tmp, d->cinert[i], d->cvel[i]);
        mju_crossForce(tmp1, d->cvel[i], tmp);
        for (int c = 0; c < 6; c++) {
            cfrc[i][c] += tmp1[c];
        }
    }
    for (int i = m->nbody - 1; i > 0; i--) {
        int p = m->body_parentid[i];
        if (p > 0) {
            for (int c = 0; c < 6; c++) {
                cfrc[p][c] += cfrc[i][c];
            }
        }
    }
    for (int i = 0; i < m->nv; i++) {
        d->qfrc_bias[i] = mju_dot6(d->cdof[i], cfrc[m->dof_bodyid[i]]);
    }
}

// Joint springs and dof dampers
MJ_HD void mj_passive(const MjModel* m, MjData* d) {
    memset(d->qfrc_passive, 0, sizeof(d->qfrc_passive));
    for (int j = 0; j < m->njnt; j++) {
        float k = m->jnt_stiffness[j];
        if (k == 0.0f) {
            continue;
        }
        int padr = m->jnt_qposadr[j];
        int dadr = m->jnt_dofadr[j];
        assert(m->jnt_type[j] >= MJ_JNT_SLIDE && "free/ball joint springs not supported");
        d->qfrc_passive[dadr] -= k*(d->qpos[padr] - m->qpos_spring[padr]);
    }
    for (int i = 0; i < m->nv; i++) {
        d->qfrc_passive[i] -= m->dof_damping[i]*d->qvel[i];
    }
}

// Motor actuators on hinge/slide joints: force = gear * clamped ctrl
MJ_HD void mj_actuation(const MjModel* m, MjData* d) {
    memset(d->qfrc_actuator, 0, sizeof(d->qfrc_actuator));
    for (int u = 0; u < m->nu; u++) {
        float ctrl = d->ctrl[u];
        if (m->actuator_ctrllimited[u]) {
            ctrl = fminf(fmaxf(ctrl, m->actuator_ctrlrange[u][0]), m->actuator_ctrlrange[u][1]);
        }
        d->qfrc_actuator[m->jnt_dofadr[m->actuator_trnid[u]]] += m->actuator_gear[u]*ctrl;
    }
}

// Collision detection: geom pair filtering and primitive colliders

typedef struct {
    float dist;
    float pos[3];
    float normal[3];
    float tangent[3];
} MjPreContact;

MJ_HD int mjraw_PlaneSphere(MjPreContact* con, float margin, const float* pos1,
    const float* mat1, const float* pos2, float radius) {
    con->normal[0] = mat1[2];
    con->normal[1] = mat1[5];
    con->normal[2] = mat1[8];
    float tmp[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
    float cdist = mju_dot3(tmp, con->normal);
    if (cdist > margin + radius) {
        return 0;
    }
    con->dist = cdist - radius;
    float s = -0.5f*con->dist - radius;
    for (int k = 0; k < 3; k++) {
        con->pos[k] = pos2[k] + s*con->normal[k];
        con->tangent[k] = 0.0f;
    }
    return 1;
}

MJ_HD int mjraw_SphereSphere(MjPreContact* con, float margin, const float* pos1,
    const float* mat1, float rad1, const float* pos2, const float* mat2, float rad2) {
    float dif[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};
    float cdist_sqr = mju_dot3(dif, dif);
    float min_dist = margin + rad1 + rad2;
    if (cdist_sqr > min_dist*min_dist) {
        return 0;
    }
    con->dist = sqrtf(cdist_sqr) - rad1 - rad2;
    for (int k = 0; k < 3; k++) {
        con->normal[k] = pos2[k] - pos1[k];
    }
    if (mju_normalize3(con->normal) < MJ_MINVAL) {
        float axis1[3] = {mat1[2], mat1[5], mat1[8]};
        float axis2[3] = {mat2[2], mat2[5], mat2[8]};
        mju_cross(con->normal, axis1, axis2);
        mju_normalize3(con->normal);
    }
    float s = rad1 + 0.5f*con->dist;
    for (int k = 0; k < 3; k++) {
        con->pos[k] = pos1[k] + s*con->normal[k];
        con->tangent[k] = 0.0f;
    }
    return 1;
}

MJ_HD int mjc_CapsuleCapsule(MjPreContact* con, float margin, const float* pos1,
    const float* mat1, const float* size1, const float* pos2, const float* mat2,
    const float* size2) {
    float axis1[3] = {mat1[2]*size1[1], mat1[5]*size1[1], mat1[8]*size1[1]};
    float axis2[3] = {mat2[2]*size2[1], mat2[5]*size2[1], mat2[8]*size2[1]};
    float dif[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};
    float ma = mju_dot3(axis1, axis1);
    float mb = -mju_dot3(axis1, axis2);
    float mc = mju_dot3(axis2, axis2);
    float u = -mju_dot3(axis1, dif);
    float v = mju_dot3(axis2, dif);
    float det = ma*mc - mb*mb;
    float vec1[3], vec2[3];
    if (fabsf(det) >= MJ_MINVAL) {
        float x1 = (mc*u - mb*v) / det;
        float x2 = (ma*v - mb*u) / det;
        if (x1 > 1.0f) {
            x1 = 1.0f;
            x2 = (v - mb) / mc;
        } else if (x1 < -1.0f) {
            x1 = -1.0f;
            x2 = (v + mb) / mc;
        }
        if (x2 > 1.0f) {
            x2 = 1.0f;
            x1 = fminf(fmaxf((u - mb) / ma, -1.0f), 1.0f);
        } else if (x2 < -1.0f) {
            x2 = -1.0f;
            x1 = fminf(fmaxf((u + mb) / ma, -1.0f), 1.0f);
        }
        for (int k = 0; k < 3; k++) {
            vec1[k] = pos1[k] + x1*axis1[k];
            vec2[k] = pos2[k] + x2*axis2[k];
        }
        return mjraw_SphereSphere(con, margin, vec1, mat1, size1[0], vec2, mat2, size2[0]);
    }
    // parallel axes: test both ends of each capsule, stop at two contacts
    int n = 0;
    float ends[4][2] = {{1.0f, 0.0f}, {-1.0f, 0.0f}, {0.0f, 1.0f}, {0.0f, -1.0f}};
    for (int e = 0; e < 4 && n < 2; e++) {
        float x1 = ends[e][0];
        float x2 = ends[e][1];
        if (e < 2) {
            x2 = fminf(fmaxf((v - x1*mb) / mc, -1.0f), 1.0f);
        } else {
            x1 = fminf(fmaxf((u - x2*mb) / ma, -1.0f), 1.0f);
        }
        for (int k = 0; k < 3; k++) {
            vec1[k] = pos1[k] + x1*axis1[k];
            vec2[k] = pos2[k] + x2*axis2[k];
        }
        n += mjraw_SphereSphere(con + n, margin, vec1, mat1, size1[0], vec2, mat2, size2[0]);
    }
    return n;
}

MJ_HD void mj_collision(const MjModel* m, MjData* d) {
    d->ncon = 0;
    for (int ga = 0; ga < m->ngeom; ga++) {
        for (int gb = ga + 1; gb < m->ngeom; gb++) {
            int g1 = ga;
            int g2 = gb;
            if (m->geom_type[g1] > m->geom_type[g2]) {
                g1 = gb;
                g2 = ga;
            }
            int w1 = m->body_weldid[m->geom_bodyid[g1]];
            int w2 = m->body_weldid[m->geom_bodyid[g2]];
            if (w1 == w2 || (m->body_dofnum[w1] == 0 && m->body_dofnum[w2] == 0)) {
                continue;
            }
            if (w1 && w2 && (w1 == m->body_weldid[m->body_parentid[w2]]
                    || w2 == m->body_weldid[m->body_parentid[w1]])) {
                continue;
            }
            if (!(m->geom_contype[g1] & m->geom_conaffinity[g2])
                    && !(m->geom_contype[g2] & m->geom_conaffinity[g1])) {
                continue;
            }
            float margin = m->geom_margin[g1] + m->geom_margin[g2];
            float bound = margin + m->geom_gap[g1] + m->geom_gap[g2];
            int t1 = m->geom_type[g1];
            int t2 = m->geom_type[g2];
            if (t1 == MJ_GEOM_PLANE && t2 == MJ_GEOM_PLANE) {
                continue;
            }
            float* pos1 = d->geom_xpos[g1];
            float* pos2 = d->geom_xpos[g2];
            float* mat1 = d->geom_xmat[g1];
            float* mat2 = d->geom_xmat[g2];
            const float* size1 = m->geom_size[g1];
            const float* size2 = m->geom_size[g2];
            float r1 = m->geom_rbound[g1];
            float r2 = m->geom_rbound[g2];
            if (t1 == MJ_GEOM_PLANE) {
                float n[3] = {mat1[2], mat1[5], mat1[8]};
                float dif[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
                if (mju_dot3(n, dif) > bound + r2) {
                    continue;
                }
            } else {
                float dif[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};
                if (mju_dot3(dif, dif) > (r1 + r2 + bound)*(r1 + r2 + bound)) {
                    continue;
                }
            }
            MjPreContact pre[4];
            int n = 0;
            float axis2[3] = {mat2[2], mat2[5], mat2[8]};
            if (t1 == MJ_GEOM_PLANE && t2 == MJ_GEOM_SPHERE) {
                n = mjraw_PlaneSphere(pre, margin, pos1, mat1, pos2, size2[0]);
            } else if (t1 == MJ_GEOM_PLANE && t2 == MJ_GEOM_CAPSULE) {
                // one sphere test per capsule end, frames aligned with the axis
                for (int end = 1; end >= -1; end -= 2) {
                    float p[3] = {pos2[0] + end*size2[1]*axis2[0], pos2[1] + end*size2[1]*axis2[1],
                        pos2[2] + end*size2[1]*axis2[2]};
                    if (mjraw_PlaneSphere(pre + n, margin, pos1, mat1, p, size2[0])) {
                        memcpy(pre[n].tangent, axis2, sizeof(axis2));
                        n++;
                    }
                }
            } else if (t1 == MJ_GEOM_SPHERE && t2 == MJ_GEOM_SPHERE) {
                n = mjraw_SphereSphere(pre, margin, pos1, mat1, size1[0], pos2, mat2, size2[0]);
            } else if (t1 == MJ_GEOM_SPHERE && t2 == MJ_GEOM_CAPSULE) {
                // sphere against the nearest point of the capsule segment
                float vec[3] = {pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2]};
                float x = fminf(fmaxf(mju_dot3(axis2, vec), -size2[1]), size2[1]);
                for (int k = 0; k < 3; k++) {
                    vec[k] = pos2[k] + x*axis2[k];
                }
                n = mjraw_SphereSphere(pre, margin, pos1, mat1, size1[0], vec, mat2, size2[0]);
            } else if (t1 == MJ_GEOM_CAPSULE && t2 == MJ_GEOM_CAPSULE) {
                n = mjc_CapsuleCapsule(pre, margin, pos1, mat1, size1, pos2, mat2, size2);
            } else {
                assert(0 && "unsupported geom pair");
            }
            for (int c = 0; c < n && d->ncon < MJ_MAXCON; c++) {
                MjContact* con = &d->contact[d->ncon];
                float includemargin = margin - m->geom_gap[g1] - m->geom_gap[g2];
                if (pre[c].dist >= includemargin) {
                    continue;
                }
                con->geom1 = g1;
                con->geom2 = g2;
                con->dist = pre[c].dist;
                con->includemargin = includemargin;
                memcpy(con->pos, pre[c].pos, sizeof(con->pos));
                // frame rows: normal, tangent1 (hint or a default orthogonal), tangent2
                float* frame = con->frame;
                memcpy(frame, pre[c].normal, 3*sizeof(float));
                memcpy(frame + 3, pre[c].tangent, 3*sizeof(float));
                mju_normalize3(frame);
                if (mju_dot3(frame + 3, frame + 3) < 0.25f) {
                    memset(frame + 3, 0, 3*sizeof(float));
                    frame[frame[1] < 0.5f && frame[1] > -0.5f ? 4 : 5] = 1.0f;
                }
                float dot = mju_dot3(frame, frame + 3);
                for (int k = 0; k < 3; k++) {
                    frame[3 + k] -= dot*frame[k];
                }
                mju_normalize3(frame + 3);
                mju_cross(frame + 6, frame, frame + 3);
                // mix geom parameters: priority, else solmix blend, max friction/condim
                int p1 = m->geom_priority[g1];
                int p2 = m->geom_priority[g2];
                float fri[3];
                if (p1 != p2) {
                    int g = p1 > p2 ? g1 : g2;
                    con->dim = m->geom_condim[g];
                    memcpy(con->solref, m->geom_solref[g], sizeof(con->solref));
                    memcpy(con->solimp, m->geom_solimp[g], sizeof(con->solimp));
                    memcpy(fri, m->geom_friction[g], sizeof(fri));
                } else {
                    int c1 = m->geom_condim[g1];
                    int c2 = m->geom_condim[g2];
                    con->dim = c1 > c2 ? c1 : c2;
                    float s1 = m->geom_solmix[g1];
                    float s2 = m->geom_solmix[g2];
                    float mix = 0.5f;
                    if (s1 >= MJ_MINVAL && s2 >= MJ_MINVAL) {
                        mix = s1 / (s1 + s2);
                    } else if (s1 >= MJ_MINVAL || s2 >= MJ_MINVAL) {
                        mix = s1 >= MJ_MINVAL ? 1.0f : 0.0f;
                    }
                    for (int k = 0; k < 2; k++) {
                        con->solref[k] = mix*m->geom_solref[g1][k]
                            + (1.0f - mix)*m->geom_solref[g2][k];
                    }
                    for (int k = 0; k < 5; k++) {
                        con->solimp[k] = mix*m->geom_solimp[g1][k]
                            + (1.0f - mix)*m->geom_solimp[g2][k];
                    }
                    for (int k = 0; k < 3; k++) {
                        fri[k] = fmaxf(m->geom_friction[g1][k], m->geom_friction[g2][k]);
                    }
                }
                con->friction[0] = fri[0];
                con->friction[1] = fri[0];
                con->friction[2] = fri[1];
                con->friction[3] = fri[2];
                con->friction[4] = fri[2];
                d->ncon++;
            }
        }
    }
}

// Constraints: rows of efc_J with pos/margin, MuJoCo's impedance and
// reference acceleration, then the dual QP for the constraint forces

// Jacobian of a world point attached to body (jacp: translation, 3 x nv rows)
MJ_HD void mj_jac(const MjModel* m, MjData* d, float jacp[3][MJ_MAX_NV], float jacr[3][MJ_MAX_NV],
    const float* point, int body) {
    memset(jacp, 0, 3*MJ_MAX_NV*sizeof(float));
    if (jacr) {
        memset(jacr, 0, 3*MJ_MAX_NV*sizeof(float));
    }
    float* com = d->subtree_com[m->body_rootid[body]];
    float offset[3] = {point[0] - com[0], point[1] - com[1], point[2] - com[2]};
    body = m->body_weldid[body];
    if (m->body_dofnum[body] == 0) {
        return;
    }
    for (int i = m->body_dofadr[body] + m->body_dofnum[body] - 1; i >= 0;
            i = m->dof_parentid[i]) {
        float* cdof = d->cdof[i];
        float tmp[3];
        mju_cross(tmp, cdof, offset);
        jacp[0][i] = cdof[3] + tmp[0];
        jacp[1][i] = cdof[4] + tmp[1];
        jacp[2][i] = cdof[5] + tmp[2];
        if (jacr) {
            jacr[0][i] = cdof[0];
            jacr[1][i] = cdof[1];
            jacr[2][i] = cdof[2];
        }
    }
}

// Regularization R = (1 - imp)/imp * diagA and reference acceleration
// aref = -b vel - k imp (pos - margin) for `size` rows starting at row i.
MJ_HD void mj_rowParams(const MjModel* m, MjData* d, int i, int size, const float* solref,
    const float* solimp, const float* diagA) {
    // solimp impedance (dmin, dmax, width, midpoint, power) of the violation
    float d0 = fminf(MJ_MAXIMP, fmaxf(MJ_MINIMP, solimp[0]));
    float dmax = fminf(MJ_MAXIMP, fmaxf(MJ_MINIMP, solimp[1]));
    float width = fmaxf(0.0f, solimp[2]);
    float mid = fminf(MJ_MAXIMP, fmaxf(MJ_MINIMP, solimp[3]));
    float power = fmaxf(1.0f, solimp[4]);
    float x = fabsf((d->efc_pos[i] - d->efc_margin[i]) / width);
    float y = x;
    if (power != 1.0f && x <= mid) {
        y = powf(x, power) / powf(mid, power - 1.0f);
    } else if (power != 1.0f && x < 1.0f) {
        y = 1.0f - powf(1.0f - x, power) / powf(1.0f - mid, power - 1.0f);
    }
    float imp = d0 + fminf(y, 1.0f)*(dmax - d0);
    if (d0 == dmax || width <= MJ_MINVAL) {
        imp = 0.5f*(d0 + dmax);
    }
    float ref0 = fmaxf(solref[0], 2.0f*m->opt_timestep);
    float k = 1.0f / fmaxf(MJ_MINVAL, dmax*dmax*ref0*ref0*solref[1]*solref[1]);
    float b = 2.0f / fmaxf(MJ_MINVAL, dmax*ref0);
    for (int j = i; j < i + size; j++) {
        d->efc_R[j] = fmaxf(MJ_MINVAL, (1.0f - imp)*diagA[j - i]/imp);
        float vel = 0.0f;
        for (int v = 0; v < m->nv; v++) {
            vel += d->efc_J[j][v]*d->qvel[v];
        }
        d->efc_aref[j] = -b*vel - k*imp*(d->efc_pos[j] - d->efc_margin[j]);
    }
}

MJ_HD void mj_makeConstraint(const MjModel* m, MjData* d) {
    d->nefc = 0;
    for (int j = 0; j < m->njnt; j++) {
        if (!m->jnt_limited[j]) {
            continue;
        }
        assert(m->jnt_type[j] == MJ_JNT_HINGE || m->jnt_type[j] == MJ_JNT_SLIDE);
        float value = d->qpos[m->jnt_qposadr[j]];
        for (int side = -1; side <= 1; side += 2) {
            float dist = side*(m->jnt_range[j][(side + 1)/2] - value);
            if (dist >= m->jnt_margin[j]) {
                continue;
            }
            int i = d->nefc++;
            int dof = m->jnt_dofadr[j];
            memset(d->efc_J[i], 0, MJ_MAX_NV*sizeof(float));
            d->efc_J[i][dof] = -side;
            d->efc_pos[i] = dist;
            d->efc_margin[i] = m->jnt_margin[j];
            float diagA = m->dof_invweight0[dof];
            mj_rowParams(m, d, i, 1, m->jnt_solref[j], m->jnt_solimp[j], &diagA);
        }
    }
    for (int c = 0; c < d->ncon; c++) {
        MjContact* con = &d->contact[c];
        int b1 = m->geom_bodyid[con->geom1];
        int b2 = m->geom_bodyid[con->geom2];
        int dim = con->dim;
        int rows = dim == 1 ? 1 : 2*(dim - 1);
        con->efc_address = -1;
        if (d->nefc + rows > MJ_MAXEFC) {
            break;
        }
        // Jacobian difference (body2 - body1) at the contact point, rotated
        // into the contact frame: rows normal, tangent1, tangent2, and for
        // condim > 3 the rotational rows torsion, roll1, roll2
        float jac1[3][MJ_MAX_NV], jac2[3][MJ_MAX_NV], jac[6][MJ_MAX_NV];
        float jacr1[3][MJ_MAX_NV], jacr2[3][MJ_MAX_NV];
        mj_jac(m, d, jac1, dim > 3 ? jacr1 : NULL, con->pos, b1);
        mj_jac(m, d, jac2, dim > 3 ? jacr2 : NULL, con->pos, b2);
        for (int r = 0; r < 3; r++) {
            for (int v = 0; v < m->nv; v++) {
                float dp = jac2[0][v] - jac1[0][v];
                float dq = jac2[1][v] - jac1[1][v];
                float dr = jac2[2][v] - jac1[2][v];
                jac[r][v] = con->frame[3*r]*dp + con->frame[3*r + 1]*dq + con->frame[3*r + 2]*dr;
                if (dim > 3) {
                    dp = jacr2[0][v] - jacr1[0][v];
                    dq = jacr2[1][v] - jacr1[1][v];
                    dr = jacr2[2][v] - jacr1[2][v];
                    jac[3 + r][v] = con->frame[3*r]*dp + con->frame[3*r + 1]*dq
                        + con->frame[3*r + 2]*dr;
                }
            }
        }
        int i0 = d->nefc;
        con->efc_address = i0;
        float tran = m->body_invweight0[b1][0] + m->body_invweight0[b2][0];
        float rot = m->body_invweight0[b1][1] + m->body_invweight0[b2][1];
        float diagA[10];
        if (dim == 1) {
            memcpy(d->efc_J[i0], jac[0], MJ_MAX_NV*sizeof(float));
            diagA[0] = tran;
        } else {
            for (int k = 1; k < dim; k++) {
                float fri = con->friction[k - 1];
                for (int v = 0; v < m->nv; v++) {
                    d->efc_J[i0 + 2*(k - 1)][v] = jac[0][v] + fri*jac[k][v];
                    d->efc_J[i0 + 2*(k - 1) + 1][v] = jac[0][v] - fri*jac[k][v];
                }
                diagA[2*(k - 1)] = tran + fri*fri*(k < 3 ? tran : rot);
                diagA[2*(k - 1) + 1] = diagA[2*(k - 1)];
            }
        }
        for (int i = i0; i < i0 + rows; i++) {
            d->efc_pos[i] = con->dist;
            d->efc_margin[i] = con->includemargin;
        }
        d->nefc += rows;
        mj_rowParams(m, d, i0, rows, con->solref, con->solimp, diagA);
        if (dim > 1) {
            // pyramidal cone: common R matching the friction impedance of the
            // elliptic model, R1 = R0 / impratio
            float r1 = d->efc_R[i0] / fmaxf(MJ_MINVAL, m->opt_impratio);
            float mu = con->friction[0]*sqrtf(r1 / d->efc_R[i0]);
            float rpy = 2.0f*mu*mu*d->efc_R[i0];
            for (int i = i0; i < i0 + rows; i++) {
                d->efc_R[i] = rpy;
            }
        }
    }
}

// Constraint forces: min 1/2 f^T (A + R) f - f^T (aref - J qacc_smooth) with
// f >= 0 and A = J M^-1 J^T, by active-set (Lawson-Hanson NNLS) on the free
// set. Sets efc_force, qfrc_constraint and qacc. The dense matrices are the
// scratch bound by mj_makeData: efc_AR = A + R, efc_ARfree its factored
// free-set block and efc_MinvJT = M^-1 J^T, indexed with stride s.
MJ_HD void mj_solveConstraint(const MjModel* m, MjData* d) {
    int nefc = d->nefc;
    int s = d->efc_stride;
    memcpy(d->qacc, d->qacc_smooth, sizeof(d->qacc));
    memset(d->qfrc_constraint, 0, sizeof(d->qfrc_constraint));
    if (nefc == 0) {
        return;
    }
    float* MinvJT = d->efc_MinvJT;
    float* G = d->efc_AR;
    float* Gp = d->efc_ARfree;
    float b[MJ_MAXEFC], z[MJ_MAXEFC], dqacc[MJ_MAX_NV] = {0};
    float* fc = d->efc_force;
    int isfree[MJ_MAXEFC], idx[MJ_MAXEFC];
    for (int i = 0; i < nefc; i++) {
        float row[MJ_MAX_NV];
        memcpy(row, d->efc_J[i], sizeof(row));
        mju_cholSolve(&d->qLD[0][0], m->nv, MJ_MAX_NV, 1, row);
        b[i] = d->efc_aref[i];
        for (int v = 0; v < m->nv; v++) {
            MinvJT[(i*MJ_MAX_NV + v)*s] = row[v];
            b[i] -= d->efc_J[i][v]*d->qacc_smooth[v];
        }
        for (int j = 0; j <= i; j++) {
            float g = 0.0f;
            for (int v = 0; v < m->nv; v++) {
                g += d->efc_J[i][v]*MinvJT[(j*MJ_MAX_NV + v)*s];
            }
            G[(i*MJ_MAXEFC + j)*s] = g;
            G[(j*MJ_MAXEFC + i)*s] = g;
        }
        G[(i*MJ_MAXEFC + i)*s] += d->efc_R[i];
        fc[i] = 0.0f;
        isfree[i] = 0;
    }
    for (int it = 0; it < 3*nefc + 10; it++) {
        // most violated bound row (negative gradient) joins the free set
        int best = -1;
        float gbest = 0.0f;
        for (int i = 0; i < nefc; i++) {
            if (isfree[i]) {
                continue;
            }
            float g = -b[i];
            for (int v = 0; v < m->nv; v++) {
                g += d->efc_J[i][v]*dqacc[v];
            }
            if (g < gbest - 1e-3f*(1.0f + 0.01f*fabsf(b[i]))) {
                gbest = g;
                best = i;
            }
        }
        if (best < 0) {
            break;
        }
        isfree[best] = 1;
        // solve on the free set; step until a free row hits zero and drop it
        for (;;) {
            int np = 0;
            for (int i = 0; i < nefc; i++) {
                if (isfree[i]) {
                    idx[np++] = i;
                }
            }
            for (int a = 0; a < np; a++) {
                z[a] = b[idx[a]];
                for (int c = 0; c < np; c++) {
                    Gp[(a*MJ_MAXEFC + c)*s] = G[(idx[a]*MJ_MAXEFC + idx[c])*s];
                }
            }
            mju_cholFactor(Gp, np, MJ_MAXEFC, s);
            mju_cholSolve(Gp, np, MJ_MAXEFC, s, z);
            float alpha = 1.0f;
            for (int a = 0; a < np; a++) {
                float fi = fc[idx[a]];
                if (z[a] <= 0.0f && fi > z[a]) {
                    alpha = fminf(alpha, fi / (fi - z[a]));
                }
            }
            for (int a = 0; a < np; a++) {
                int i = idx[a];
                float fi = fc[i];
                fc[i] = fi + alpha*(z[a] - fi);
                if (z[a] <= 0.0f && fi <= alpha*(fi - z[a])*(1.0f + 1e-5f)) {
                    fc[i] = 0.0f;
                    isfree[i] = 0;
                }
            }
            if (alpha >= 1.0f) {
                break;
            }
        }
        memset(dqacc, 0, sizeof(dqacc));
        for (int i = 0; i < nefc; i++) {
            for (int v = 0; v < m->nv; v++) {
                dqacc[v] += fc[i]*MinvJT[(i*MJ_MAX_NV + v)*s];
            }
        }
    }
    for (int i = 0; i < nefc; i++) {
        for (int v = 0; v < m->nv; v++) {
            d->qfrc_constraint[v] += fc[i]*d->efc_J[i][v];
            d->qacc[v] += fc[i]*MinvJT[(i*MJ_MAX_NV + v)*s];
        }
    }
}

// Body accelerations and interaction forces including constraint forces:
// cacc, cfrc_int and cfrc_ext (torque:force in the subtree COM frame; contact
// forces decoded from the pyramid rows)
MJ_HD void mj_rnePostConstraint(const MjModel* m, MjData* d) {
    memset(d->cfrc_ext, 0, sizeof(d->cfrc_ext));
    for (int c = 0; c < d->ncon; c++) {
        MjContact* con = &d->contact[c];
        int adr = con->efc_address;
        if (adr < 0) {
            continue;
        }
        float lfrc[6] = {0};
        if (con->dim == 1) {
            lfrc[0] = d->efc_force[adr];
        } else {
            for (int k = 0; k < con->dim - 1; k++) {
                lfrc[0] += d->efc_force[adr + 2*k] + d->efc_force[adr + 2*k + 1];
                lfrc[k + 1] = (d->efc_force[adr + 2*k] - d->efc_force[adr + 2*k + 1])
                    *con->friction[k];
            }
        }
        float cfrc[6];
        mju_mulMatTVec3(cfrc, con->frame, lfrc + 3);
        mju_mulMatTVec3(cfrc + 3, con->frame, lfrc);
        int bodies[2] = {m->geom_bodyid[con->geom1], m->geom_bodyid[con->geom2]};
        for (int side = 0; side < 2; side++) {
            int k = bodies[side];
            if (k == 0) {
                continue;
            }
            float* com = d->subtree_com[m->body_rootid[k]];
            float dif[3] = {com[0] - con->pos[0], com[1] - con->pos[1], com[2] - con->pos[2]};
            float cros[3];
            mju_cross(cros, dif, cfrc + 3);
            float sign = side ? 1.0f : -1.0f;
            for (int i = 0; i < 3; i++) {
                d->cfrc_ext[k][i] += sign*(cfrc[i] - cros[i]);
                d->cfrc_ext[k][i + 3] += sign*cfrc[i + 3];
            }
        }
    }
    memset(d->cacc[0], 0, 6*sizeof(float));
    memset(d->cfrc_int[0], 0, 6*sizeof(float));
    d->cacc[0][3] = -m->opt_gravity[0];
    d->cacc[0][4] = -m->opt_gravity[1];
    d->cacc[0][5] = -m->opt_gravity[2];
    for (int j = 1; j < m->nbody; j++) {
        int bda = m->body_dofadr[j];
        float cfrc_body[6], cfrc_corr[6], cfrc[6];
        memcpy(d->cacc[j], d->cacc[m->body_parentid[j]], 6*sizeof(float));
        for (int i = bda; i < bda + m->body_dofnum[j]; i++) {
            for (int c = 0; c < 6; c++) {
                d->cacc[j][c] += d->cdof_dot[i][c]*d->qvel[i] + d->cdof[i][c]*d->qacc[i];
            }
        }
        mju_mulInertVec(cfrc_body, d->cinert[j], d->cacc[j]);
        mju_mulInertVec(cfrc_corr, d->cinert[j], d->cvel[j]);
        mju_crossForce(cfrc, d->cvel[j], cfrc_corr);
        for (int c = 0; c < 6; c++) {
            d->cfrc_int[j][c] = cfrc_body[c] + cfrc[c] - d->cfrc_ext[j][c];
        }
    }
    for (int j = m->nbody - 1; j > 0; j--) {
        for (int c = 0; c < 6; c++) {
            d->cfrc_int[m->body_parentid[j]][c] += d->cfrc_int[j][c];
        }
    }
}

// Forward dynamics: qacc and all intermediate quantities from qpos, qvel, ctrl
MJ_HD void mj_forward(const MjModel* m, MjData* d) {
    mj_kinematics(m, d);
    mj_comPos(m, d);
    mj_crb(m, d);
    mj_collision(m, d);
    mj_comVel(m, d);
    mj_passive(m, d);
    mj_makeConstraint(m, d);
    mj_rne(m, d);
    mj_actuation(m, d);
    for (int i = 0; i < m->nv; i++) {
        d->qfrc_smooth[i] = d->qfrc_passive[i] - d->qfrc_bias[i] + d->qfrc_actuator[i];
    }
    memcpy(d->qacc_smooth, d->qfrc_smooth, sizeof(d->qacc_smooth));
    mju_cholSolve(&d->qLD[0][0], m->nv, MJ_MAX_NV, 1, d->qacc_smooth);
    mj_solveConstraint(m, d);
}

MJ_HD void mj_integratePos(const MjModel* m, float* qpos, const float* qvel, float dt) {
    for (int j = 0; j < m->njnt; j++) {
        int padr = m->jnt_qposadr[j];
        int vadr = m->jnt_dofadr[j];
        int jtype = m->jnt_type[j];
        if (jtype == MJ_JNT_FREE) {
            for (int k = 0; k < 3; k++) {
                qpos[padr + k] += dt*qvel[vadr + k];
            }
            mju_quatIntegrate(qpos + padr + 3, qvel + vadr + 3, dt);
        } else if (jtype == MJ_JNT_BALL) {
            mju_quatIntegrate(qpos + padr, qvel + vadr, dt);
        } else {
            qpos[padr] += dt*qvel[vadr];
        }
    }
}

// Semi-implicit Euler with implicit joint damping: (M + h B) qacc' = M qacc
MJ_HD void mj_Euler(const MjModel* m, MjData* d) {
    float h = m->opt_timestep;
    float qacc[MJ_MAX_NV];
    float MhB[MJ_MAX_NV][MJ_MAX_NV];
    memcpy(MhB, d->qM, sizeof(MhB));
    for (int i = 0; i < m->nv; i++) {
        MhB[i][i] += h*m->dof_damping[i];
        qacc[i] = d->qfrc_smooth[i] + d->qfrc_constraint[i];
    }
    mju_cholFactor(&MhB[0][0], m->nv, MJ_MAX_NV, 1);
    mju_cholSolve(&MhB[0][0], m->nv, MJ_MAX_NV, 1, qacc);
    for (int i = 0; i < m->nv; i++) {
        d->qvel[i] += h*qacc[i];
    }
    mj_integratePos(m, d->qpos, d->qvel, h);
    d->time += h;
}

// Explicit RK4 (mj_RungeKutta with N=4)
MJ_HD void mj_RungeKutta4(const MjModel* m, MjData* d) {
    float h = m->opt_timestep;
    float A[3] = {0.5f, 0.5f, 1.0f};
    float B[4] = {1.0f/6.0f, 1.0f/3.0f, 1.0f/3.0f, 1.0f/6.0f};
    float X[4][MJ_MAX_NQ + MJ_MAX_NV], F[4][MJ_MAX_NV];
    float qpos0[MJ_MAX_NQ], time0 = d->time;
    memcpy(qpos0, d->qpos, sizeof(qpos0));
    memcpy(X[0], d->qpos, sizeof(qpos0));
    memcpy(X[0] + m->nq, d->qvel, sizeof(d->qvel));
    memcpy(F[0], d->qacc, sizeof(d->qacc));
    for (int i = 1; i < 4; i++) {
        // stage i uses only stage i-1 with weight A[i-1]
        float dv[MJ_MAX_NV], da[MJ_MAX_NV];
        for (int v = 0; v < m->nv; v++) {
            dv[v] = A[i - 1]*X[i - 1][m->nq + v];
            da[v] = A[i - 1]*F[i - 1][v];
        }
        memcpy(d->qpos, qpos0, sizeof(qpos0));
        mj_integratePos(m, d->qpos, dv, h);
        for (int v = 0; v < m->nv; v++) {
            d->qvel[v] = X[0][m->nq + v] + h*da[v];
        }
        mj_forward(m, d);
        memcpy(X[i], d->qpos, sizeof(qpos0));
        memcpy(X[i] + m->nq, d->qvel, sizeof(d->qvel));
        memcpy(F[i], d->qacc, sizeof(d->qacc));
    }
    float dv[MJ_MAX_NV], da[MJ_MAX_NV];
    for (int v = 0; v < m->nv; v++) {
        dv[v] = 0.0f;
        da[v] = 0.0f;
        for (int j = 0; j < 4; j++) {
            dv[v] += B[j]*X[j][m->nq + v];
            da[v] += B[j]*F[j][v];
        }
    }
    memcpy(d->qpos, qpos0, sizeof(qpos0));
    for (int v = 0; v < m->nv; v++) {
        d->qvel[v] = X[0][m->nq + v] + h*da[v];
    }
    mj_integratePos(m, d->qpos, dv, h);
    d->time = time0 + h;
}

MJ_HD void mj_step(const MjModel* m, MjData* d) {
    mj_forward(m, d);
    if (m->opt_integrator == MJ_INT_RK4) {
        mj_RungeKutta4(m, d);
    } else {
        assert(m->opt_integrator == MJ_INT_EULER);
        mj_Euler(m, d);
    }
}

MJ_HD void mj_resetData(const MjModel* m, MjData* d) {
    memset(&d->time, 0, sizeof(MjData) - offsetof(MjData, time));
    memcpy(d->qpos, m->qpos0, sizeof(d->qpos));
}
