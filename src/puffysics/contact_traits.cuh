#pragma once
#include "model.cuh"
#include "integrate.cuh"

struct PfSatShape { int type; PfPose pose; PfVec3 half_extents; };
struct PfSatContact {
    int hit, iterations;
    float separation;
    PfVec3 normal, point_a, point_b;
};
struct PfSatSweep { int hit, iterations; float toi; PfSatContact contact; };
struct PfContactTraits {
    using Vec3 = PfVec3;
    using Shape = PfSatShape;
    using Contact = PfSatContact;
    using Body = PfBody;
    using Sweep = PfSatSweep;
    static constexpr int box_kind = PF_BOX, sphere_kind = PF_SPHERE;
    __host__ __device__ static float min(float a, float b) { return a < b ? a : b; }
    __host__ __device__ static float max(float a, float b) { return a > b ? a : b; }
    __host__ __device__ static float clamp(float v, float lo, float hi) { return min(max(v,lo),hi); }
    __host__ __device__ static Vec3 v3(float x,float y,float z) { return pf_v3(x,y,z); }
    __host__ __device__ static Vec3 add(Vec3 a,Vec3 b) { return pf_add(a,b); }
    __host__ __device__ static Vec3 sub(Vec3 a,Vec3 b) { return pf_sub(a,b); }
    __host__ __device__ static Vec3 scale(Vec3 v,float s) { return pf_scale(v,s); }
    __host__ __device__ static Vec3 lerp(Vec3 a,Vec3 b,float t) { return add(a,scale(sub(b,a),t)); }
    __host__ __device__ static float length(Vec3 v) { return pf_length(v); }
    __host__ __device__ static float radius(const Shape* s) {
        return s->type==sphere_kind?s->half_extents.x:length(s->half_extents);
    }
    __host__ __device__ static float box_radius(const Vec3 axes[3],Vec3 half,Vec3 direction) {
        return half.x*fabsf(dot(axes[0],direction)) + half.y*fabsf(dot(axes[1],direction))
            + half.z*fabsf(dot(axes[2],direction));
    }
    __host__ __device__ static float dot(Vec3 a,Vec3 b) { return pf_dot(a,b); }
    __host__ __device__ static Vec3 cross(Vec3 a,Vec3 b) { return pf_cross(a,b); }
    __host__ __device__ static void axes(PfQuat q,Vec3* a) { pf_quat_axes(q,a); }
    __host__ __device__ static Vec3 rotate(PfQuat q,Vec3 v) { return pf_quat_rotate(q,v); }
    __host__ __device__ static PfQuat conjugate(PfQuat q) { return pf_quat_conjugate(q); }
    __device__ static PfQuat integrate_rotation(PfQuat q,Vec3 v,float dt) {
        return pf_quat_normalize(pf_quat_multiply(pf_orientation_delta(v,dt),q));
    }
    __host__ __device__ static Vec3& position(Body& b) { return b.position; }
    __host__ __device__ static const Vec3& position(const Body& b) { return b.position; }
    __host__ __device__ static PfQuat& rotation(Body& b) { return b.rotation; }
    __host__ __device__ static const PfQuat& rotation(const Body& b) { return b.rotation; }
    __host__ __device__ static bool dynamic(const Body& b) { return pf_mode_is_dynamic(b.mode); }
    __host__ __device__ static float inverse_mass(const Body& b) { return dynamic(b) ? b.inverse_mass : 0; }
    __host__ __device__ static Vec3 inverse_inertia(const Body* b,Vec3 v) { return pf_inverse_inertia_world(b,v); }
    __host__ __device__ static void apply_impulse(Body* b,Vec3 p,Vec3 v) { pf_apply_impulse(b,p,v); }
    __host__ __device__ static float impulse_denominator(const Body* b,Vec3 p,Vec3 d) {
        if (!dynamic(*b)) return 0;
        Vec3 lever=cross(sub(p,b->position),d);
        return b->inverse_mass+dot(lever,inverse_inertia(b,lever));
    }
};

// Custom policies must expose these types/methods. Reaction carries active and
// inverse_mass[3]; AngularReaction carries active and inverse_mass. Additional
// Jacobians, responses and coordinate counts belong to the custom policy.
struct PfNoContactReaction {
    static constexpr bool require_angular_reaction = false;
    struct State {};
    struct Reaction { int active; float inverse_mass[3]; };
    struct AngularReaction { int active; float inverse_mass; };
    struct Displacement {};
    __device__ static void apply(const Reaction&,State&,int,float) {}
    __device__ static float velocity(const Reaction&,const State&,int) { return 0; }
    __device__ static void apply_angular(const AngularReaction&,State&,float) {}
    __device__ static float angular_velocity(const AngularReaction&,const State&) { return 0; }
    __device__ static Displacement correct(State&,const Reaction&,float) { return {}; }
    __device__ static float displacement(const Reaction&,const Displacement&) { return 0; }
};
