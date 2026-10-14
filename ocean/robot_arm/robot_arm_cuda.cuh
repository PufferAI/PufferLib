#pragma once

#include <assert.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "robot_arm.h"

#include "../../src/puffysics/sat_manifold.cuh"
#include "../../src/puffysics/impulse_solver.cuh"
#include "../../src/puffysics/compound_contact.cuh"
#include "../../src/puffysics/swept_collision.cuh"

// Preserve the environment's xyzw layout and arithmetic during extraction.
// The engine's default traits operate directly on PfBody and wxyz quaternions.
struct RaContactTraits {
    using Vec3 = RaVec3;
    using Shape = RaConvexShape;
    using Contact = RaConvexContact;
    using Body = RaRigidBody;
    using Sweep = RaConvexSweep;
    static constexpr int box_kind = RA_CONVEX_BOX, sphere_kind = RA_CONVEX_SPHERE;
    RA_HD static RA_INLINE float min(float a, float b) { return ra_min(a, b); }
    RA_HD static RA_INLINE float max(float a, float b) { return ra_max(a, b); }
    RA_HD static RA_INLINE float clamp(float v, float lo, float hi) { return ra_clamp(v, lo, hi); }
    RA_HD static RA_INLINE Vec3 v3(float x, float y, float z) { return ra_v3(x, y, z); }
    RA_HD static RA_INLINE Vec3 add(Vec3 a, Vec3 b) { return ra_add(a, b); }
    RA_HD static RA_INLINE Vec3 sub(Vec3 a, Vec3 b) { return ra_sub(a, b); }
    RA_HD static RA_INLINE Vec3 scale(Vec3 v, float s) { return ra_scale(v, s); }
    RA_HD static RA_INLINE Vec3 lerp(Vec3 a, Vec3 b, float t) {
        return ra_add(a, ra_scale(ra_sub(b, a), t));
    }
    RA_HD static RA_INLINE float length(Vec3 v) { return ra_length(v); }
    RA_D static RA_INLINE float radius(const Shape* s) { return ra_brad(s); }
    RA_D static RA_INLINE float box_radius(const Vec3 axes[3], Vec3 half, Vec3 d) {
        return half.x * fabsf(ra_dot(axes[0], d)) + half.y * fabsf(ra_dot(axes[1], d))
            + half.z * fabsf(ra_dot(axes[2], d));
    }
    RA_HD static RA_INLINE float dot(Vec3 a, Vec3 b) { return ra_dot(a, b); }
    RA_HD static RA_INLINE Vec3 cross(Vec3 a, Vec3 b) { return ra_cross(a, b); }
    RA_HD static RA_INLINE Vec3 rotate(RaQuat q, Vec3 v) { return ra_rotate(q, v); }
    RA_HD static RA_INLINE RaQuat conjugate(RaQuat q) { return ra_qconj(q); }
    RA_D static RA_INLINE void axes(RaQuat q, Vec3* a) { ra_caxes(q, a); }
    RA_D static RA_INLINE RaQuat integrate_rotation(RaQuat q, Vec3 v, float dt) {
        return ra_qint(q, v, dt);
    }
    RA_HD static RA_INLINE Vec3& position(Body& b) { return b.pose.position; }
    RA_HD static RA_INLINE const Vec3& position(const Body& b) { return b.pose.position; }
    RA_HD static RA_INLINE RaQuat& rotation(Body& b) { return b.pose.rotation; }
    RA_HD static RA_INLINE const RaQuat& rotation(const Body& b) { return b.pose.rotation; }
    RA_D static RA_INLINE bool dynamic(const Body& b) { return b.mass > 0; }
    RA_D static RA_INLINE float inverse_mass(const Body& b) {
        return 1.0f / ra_max(b.mass, 1.0e-8f);
    }
    RA_D static RA_INLINE Vec3 inverse_inertia(const Body* b, Vec3 v) { return ra_invi(b, v); }
    RA_D static RA_INLINE void apply_impulse(Body* b, Vec3 p, Vec3 v) { ra_impa(b, p, v); }
    RA_D static RA_INLINE float impulse_denominator(const Body* b, Vec3 p, Vec3 d) {
        return ra_impd(b, p, d);
    }
};

typedef struct PlImpulseReaction {
    int active; // mass==0 proxy; live qd/jaw, not cached body_b velocity
    float inverse_mass[3];
    float jaw_velocity_response[3];
    float robot_jacobian[3][RA_DOF];
    float jaw_jacobian[3];
    float robot_response[3][RA_DOF];
} PlImpulseReaction;

typedef struct PlImpulseAngularReaction {
    int active;
    float inverse_mass;
    float robot_jacobian[RA_DOF];
    float robot_response[RA_DOF];
} PlImpulseAngularReaction;

struct RaContactReaction {
    static constexpr bool require_angular_reaction = true;
    using State = RaState;
    using Reaction = PlImpulseReaction;
    using AngularReaction = PlImpulseAngularReaction;
    struct Displacement {
        float q[RA_DOF];
        float width;
    };
    RA_D static RA_INLINE void apply(const Reaction& r, State& s, int d, float impulse) {
        ra_applyr(&s, r.robot_response[d], -impulse);
        s.gripper_velocity += r.jaw_velocity_response[d] * impulse;
    }
    RA_D static RA_INLINE float velocity(const Reaction& r, const State& s, int d) {
        float v = r.jaw_jacobian[d] * s.gripper_velocity;
        for (int j = 0; j < RA_DOF; ++j) {
            v += r.robot_jacobian[d][j] * s.qd[j];
        }
        return v;
    }
    RA_D static RA_INLINE void apply_angular(const AngularReaction& r, State& s, float moment) {
        ra_applyr(&s, r.robot_response, -moment);
    }
    RA_D static RA_INLINE float angular_velocity(const AngularReaction& r, const State& s) {
        float v = 0;
        for (int j = 0; j < RA_DOF; ++j) {
            v += r.robot_jacobian[j] * s.qd[j];
        }
        return v;
    }
    RA_D static RA_INLINE Displacement correct(State& s, const Reaction& r, float magnitude) {
        Displacement delta = {};
        for (int j = 0; j < RA_DOF; ++j) {
            delta.q[j] = -magnitude * r.robot_response[0][j];
        }
        delta.width = magnitude * r.jaw_velocity_response[0];
        // Retain the legacy split pass's velocity update during extraction.
        ra_applyr(&s, r.robot_response[0], -magnitude);
        s.gripper_width += r.jaw_velocity_response[0] * magnitude;
        s.gripper_width = ra_clamp(s.gripper_width, 0.0f, 0.20f);
        return delta;
    }
    RA_D static RA_INLINE float displacement(const Reaction& r, const Displacement& delta) {
        float d = r.jaw_jacobian[0] * delta.width;
        for (int j = 0; j < RA_DOF; ++j) {
            d += r.robot_jacobian[0][j] * delta.q[j];
        }
        return d;
    }
};
using RaSat = PfSatCollisionT<RaContactTraits>;
using RaImpulse = PfImpulseSolverT<RaContactTraits, RaContactReaction>;
using PlSatFeature = RaSat::Feature;
using PlSatObb = RaSat::Obb;
using PlSatQuery = RaSat::Query;
using PlSatManifold = RaSat::Manifold;
using PlImpulseConfig = RaImpulse::Config;
using PlImpulsePatch = RaImpulse::Patch;
using PlImpulseCandidate = RaImpulse::Candidate;
using PlImpulsePoint = RaImpulse::Point;
using PlImpulseManifold = RaImpulse::Manifold;
using PlImpulseCacheEntry = RaImpulse::CacheEntry;
using PlImpulseCache = RaImpulse::Cache;

// Keep existing cache IDs stable; generic compound geometry owns no such layout.
struct RaCompoundFeatures {
    RA_D static RA_INLINE uint32_t box(uint32_t side, int owner, int x, int z, uint32_t vertex) {
        uint32_t local =
            (vertex & 0x0000ffffu) | ((uint32_t)((x * RA_PAD_BOXES + z) & 0xffu) << 16);
        return 0xd0000000u | ((side & 1) << 27) | ((uint32_t)(owner & 7) << 24) | local;
    }
    RA_D static RA_INLINE uint32_t sphere(uint32_t side, int owner) {
        return 0x73000000u | (side << 8) | (uint32_t)owner;
    }
};
using RaCompound =
    PfCompoundContactT<RaContactTraits, RaImpulse, RA_PAD_BOXES, 128, RaCompoundFeatures>;
using RaSweep = PfSweptCollisionT<RaContactTraits>;
#define PL_IMPULSE_MAX_MANIFOLDS RaImpulse::max_manifolds
#define PL_IMPULSE_MAX_POINTS RaImpulse::max_points
#define PL_IMPULSE_MAX_CACHE RaImpulse::max_cache
#define PL_IMPULSE_EPSILON RaImpulse::epsilon
#define PL_SAT_MAX_MANIFOLD_POINTS RaSat::max_points

#define RA_CUDA_PAD_MAX_VISIBLE_CANDIDATES RaCompound::candidate_capacity
#define RA_CUDA_CONTACT_SLOP 1.0e-5f

#define RA_CUDA_BODY_CUBE 0
#define RA_CUDA_BODY_BASE 1
#define RA_CUDA_BODY_TABLE 2
#define RA_CUDA_BODY_SHELL_START 3
#define RA_CUDA_SHELL_BOXES 5
#define RA_CUDA_BODY_LINK_START (RA_CUDA_BODY_SHELL_START + RA_CUDA_SHELL_BOXES)
#define RA_CUDA_BODY_PAD_LEFT_START (RA_CUDA_BODY_LINK_START + RA_DOF)
#define RA_CUDA_BODY_PAD_RIGHT_START (RA_CUDA_BODY_PAD_LEFT_START + RA_PAD_BOXES)
#define RA_CUDA_ROBOT_BODY_END (RA_CUDA_BODY_PAD_RIGHT_START + RA_PAD_BOXES)
#define RA_CUDA_BODY_RIM RA_CUDA_ROBOT_BODY_END
#define RA_CUDA_BODY_BACKBOARD (RA_CUDA_BODY_RIM + 1)
#define RA_CUDA_BODIES (RA_CUDA_BODY_BACKBOARD + 1)

typedef struct RaCudaRigidWorld {
    int body_count;
    int shape_count;
    int manifold_count;
    unsigned int topology;
    uint32_t compound_pad_component_mask[2];
    RaRigidBody bodies[RA_CUDA_BODIES];
    RaConvexShape shapes[RA_CUDA_BODIES];
    PlImpulseManifold manifolds[PL_IMPULSE_MAX_MANIFOLDS];
    PlImpulseCandidate compound_candidate_scratch[RA_CUDA_PAD_MAX_VISIBLE_CANDIDATES];
    PlImpulseCache cache;
    PlImpulseConfig config;
} RaCudaRigidWorld;

RA_D static RA_INLINE int ra_pad_side(int body) {
    return body >= RA_CUDA_BODY_PAD_LEFT_START && body < RA_CUDA_BODY_PAD_RIGHT_START
        ? 0
        : (body >= RA_CUDA_BODY_PAD_RIGHT_START && body < RA_CUDA_ROBOT_BODY_END ? 1 : -1);
}

RA_D static RA_INLINE int ra_manok(RaCudaRigidWorld* world) {
    return world->manifold_count < PL_IMPULSE_MAX_MANIFOLDS;
}

typedef struct RaCudaProductionStaged {
    float actions[RA_ACTIONS];
    float target_width;
    float energy;
    int first_grasp;
    int grasp_broken;
    int released;
    RaPose links[RA_LINKS];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
} RaCudaProductionStaged;

typedef struct RaCudaProductionWorld {
    RaState state;
    RaCudaRigidWorld rigid;
    RaCudaProductionStaged staged;
} RaCudaProductionWorld;

RA_HD static RA_INLINE unsigned int ra_topo(const RaState* state) {
    return state->basketball_mode ? 3u : (state->stack_mode ? 2u : 1u);
}

RA_HD static RA_INLINE void ra_rbrst(RaCudaRigidWorld* world, unsigned int topology) {
    memset(world, 0, sizeof(*world));
    world->topology = topology;
    world->config = (PlImpulseConfig){
        .velocity_iterations = RA_CONTACT_VELOCITY_ITERS,
        .position_iterations = RA_CONTACT_POSITION_ITERS,
        .velocity_impulse_tolerance = 1.0e-7f,
        .warm_start = 1,
        .split_position = 1,
        .position_beta = 0.80f,
        .slop = RA_CUDA_CONTACT_SLOP,
        .speculative_margin = RA_CONTACT_MARGIN,
        .static_friction = 0.80f,
        .dynamic_friction = 0.70f,
        .restitution = 0.12f,
        .restitution_threshold = RA_RESTITUTION_THRESHOLD,
        .max_normal_impulse = 1.0e4f,
        .max_position_correction = 0.010f,
        .max_position_impulse = 1.0e4f,
        .cache_max_age = 24,
    };
    RaImpulse::clear_cache(&world->cache);
}

RA_D static RA_INLINE void ra_rbind(
    RaCudaRigidWorld* world, int index, RaRigidBody body, RaConvexShape shape) {
    assert(index >= 0 && index < RA_CUDA_BODIES);
    world->bodies[index] = body;
    world->shapes[index] = shape;
    int next = index + 1;
    if (next > world->body_count) {
        world->body_count = next;
    }
    if (next > world->shape_count) {
        world->shape_count = next;
    }
}

RA_D static RA_INLINE void ra_setbox(RaCudaRigidWorld* world, int index, RaPose pose,
    RaVec3 half_extents, float mass, RaVec3 linear_velocity, RaVec3 angular_velocity) {
    RaInertia3 inertia = RaExplicit::box_inertia<RaInertia3>(mass, half_extents);
    ra_rbind(world, index, (RaRigidBody){pose, linear_velocity, angular_velocity, mass, inertia},
        (RaConvexShape){RA_CONVEX_BOX, pose, half_extents});
}

RA_D static RA_INLINE void ra_setsph(RaCudaRigidWorld* world, int index, RaPose pose, float radius,
    float mass, RaVec3 linear_velocity, RaVec3 angular_velocity) {
    RaInertia3 inertia = RaExplicit::sphere_inertia<RaInertia3>(mass, radius);
    ra_rbind(world, index, (RaRigidBody){pose, linear_velocity, angular_velocity, mass, inertia},
        (RaConvexShape){RA_CONVEX_SPHERE, pose, ra_v3(radius, radius, radius)});
}

RA_D static RA_INLINE int ra_pair(RaCudaRigidWorld* world, int body_a, int body_b, float margin,
    float static_friction, float dynamic_friction, float restitution) {
    assert(body_a >= 0 && body_b >= 0 && body_a != body_b);
    assert(body_a < world->shape_count && body_b < world->shape_count);
    PlSatManifold sat;
    int candidate_count =
        RaSat::manifold(&world->shapes[body_a], &world->shapes[body_b], margin, &sat);
    if (candidate_count <= 0 || !ra_manok(world)) {
        return 0;
    }
    candidate_count = ra_min(candidate_count, PL_SAT_MAX_MANIFOLD_POINTS);
    PlImpulseCandidate candidates[PL_SAT_MAX_MANIFOLD_POINTS];
    for (int index = 0; index < candidate_count; ++index) {
        memset(&candidates[index], 0, sizeof(candidates[index]));
        candidates[index].contact = sat.point[index];
        candidates[index].feature = sat.point_feature[index];
    }
    PlImpulseManifold* manifold = &world->manifolds[world->manifold_count];
    int made_count = RaImpulse::manifold(body_a, body_b, candidates, candidate_count, margin,
        static_friction, dynamic_friction, restitution, manifold);
    world->manifold_count += made_count > 0;
    return made_count > 0;
}

RA_D static RA_INLINE RaSweep::Ring ra_ring(void) {
    return {ra_hoop(), ra_v3(1, 0, 0), ra_v3(0, 0, 1), ra_v3(0, 1, 0), RA_RIM_MAJOR_RADIUS,
        RA_RIM_TUBE_RADIUS};
}

RA_D static RA_INLINE RaConvexContact ra_rimq(RaVec3 ball_position, float margin) {
    return RaSweep::sphere_ring(ball_position, RA_BALL_RADIUS, ra_ring(), margin);
}

RA_D static RA_INLINE void ra_bodies(RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    unsigned int topology = ra_topo(state);
    if (rigid->topology != topology) {
        RaImpulse::clear_cache(&rigid->cache);
        rigid->topology = topology;
    }
    rigid->body_count = 0;
    rigid->shape_count = 0;
    rigid->manifold_count = 0;
    rigid->compound_pad_component_mask[0] = 0;
    rigid->compound_pad_component_mask[1] = 0;
    const RaPose* links = world->staged.links;
    const RaVec3* origins = world->staged.origins;
    const RaVec3* axes = world->staged.axes;
    const RaPose cube_pose = {state->cube_position, state->cube_rotation};
    const float cube_mass = state->stack_mode
        ? RA_STACK_CUBE_MASS
        : (state->basketball_mode ? RA_BALL_MASS : RA_PICK_CUBE_MASS);
    if (state->basketball_mode) {
        ra_setsph(rigid, RA_CUDA_BODY_CUBE, cube_pose, RA_BALL_RADIUS, cube_mass,
            state->cube_velocity, state->cube_angular_velocity);
    } else {
        ra_setbox(rigid, RA_CUDA_BODY_CUBE, cube_pose,
            ra_v3(RA_CUBE_HALF, RA_CUBE_HALF, RA_CUBE_HALF), cube_mass, state->cube_velocity,
            state->cube_angular_velocity);
    }
    const RaPose base_pose = {state->base_cube_position, state->base_cube_rotation};
    ra_setbox(rigid, RA_CUDA_BODY_BASE, base_pose, ra_v3(RA_CUBE_HALF, RA_CUBE_HALF, RA_CUBE_HALF),
        state->stack_mode ? RA_STACK_CUBE_MASS : 0.0f, state->base_cube_velocity,
        state->base_cube_angular_velocity);
    const RaPose table_pose = {
        ra_v3(RA_TABLE_CENTER_X, RA_TABLE_TOP - 0.5f * RA_TABLE_THICKNESS, 0), ra_quat(0, 0, 0, 1)};
    ra_setbox(rigid, RA_CUDA_BODY_TABLE, table_pose,
        ra_v3(0.5f * RA_TABLE_SIZE_X, 0.5f * RA_TABLE_THICKNESS, 0.5f * RA_TABLE_SIZE_Z), 0.0f,
        ra_v3(0, 0, 0), ra_v3(0, 0, 0));

    RaGripperCollisionFrame frame = ra_gripf(links, state->end_effector);
    const RaVec3 hand_angular = RaDynamics::angular_velocity(state->qd, axes, RA_DOF - 1);
    const int shell_source[RA_CUDA_SHELL_BOXES] = {0, 1, 2, 3, 5};
    for (int item = 0; item < RA_CUDA_SHELL_BOXES; ++item) {
        RaCollisionBox box = ra_gripb(&frame, shell_source[item]);
        RaVec3 velocity =
            RaDynamics::point_velocity(state->qd, origins, axes, RA_DOF - 1, box.pose.position);
        ra_setbox(rigid, RA_CUDA_BODY_SHELL_START + item, box.pose, box.half_extent, 0.0f, velocity,
            hand_angular);
    }
    for (int item = 0; item < RA_DOF; ++item) {
        RaCollisionBox box = ra_linkb(links, item);
        RaVec3 velocity =
            RaDynamics::point_velocity(state->qd, origins, axes, item, box.pose.position);
        RaVec3 angular = RaDynamics::angular_velocity(state->qd, axes, item);
        ra_setbox(rigid, RA_CUDA_BODY_LINK_START + item, box.pose, box.half_extent, 0.0f, velocity,
            angular);
    }
    const RaVec3 hand_axis = ra_rotate(frame.hand.rotation, ra_v3(0, 1, 0));
    for (int pad = 0; pad < RA_PAD_BOXES; ++pad) {
        for (int side = 0; side < 2; ++side) {
            RaPose finger = side == 0 ? frame.left_finger : frame.right_finger;
            RaConvexShape shape = ra_padsh(finger, pad);
            RaVec3 velocity = RaDynamics::point_velocity(
                state->qd, origins, axes, RA_DOF - 1, shape.pose.position);
            float jaw = (side == 0 ? 0.5f : -0.5f) * state->gripper_velocity;
            velocity = ra_add(velocity, ra_scale(hand_axis, jaw));
            int body =
                (side == 0 ? RA_CUDA_BODY_PAD_LEFT_START : RA_CUDA_BODY_PAD_RIGHT_START) + pad;
            ra_setbox(rigid, body, shape.pose, shape.half_extents, 0.0f, velocity, hand_angular);
        }
    }
    if (state->basketball_mode) {
        const RaPose rim_pose = {ra_hoop(), ra_quat(0, 0, 0, 1)};
        ra_setsph(rigid, RA_CUDA_BODY_RIM, rim_pose, RA_RIM_MAJOR_RADIUS + RA_RIM_TUBE_RADIUS, 0.0f,
            ra_v3(0, 0, 0), ra_v3(0, 0, 0));
        const RaPose backboard_pose = {
            ra_v3(RA_HOOP_CENTER_X, RA_BACKBOARD_CENTER_Y, RA_BACKBOARD_CENTER_Z),
            ra_quat(0, 0, 0, 1)};
        ra_setbox(rigid, RA_CUDA_BODY_BACKBOARD, backboard_pose,
            ra_v3(RA_BACKBOARD_HALF_X, RA_BACKBOARD_HALF_Y, RA_BACKBOARD_HALF_Z), 0.0f,
            ra_v3(0, 0, 0), ra_v3(0, 0, 0));
    }
}

RA_D static RA_INLINE void ra_react(
    RaCudaProductionWorld* world, const float mass_factor[RA_DOF][RA_DOF]) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    const RaPose* links = world->staged.links;
    const RaVec3* origins = world->staged.origins;
    const RaVec3* axes = world->staged.axes;
    assert(rigid->manifold_count >= 0 && rigid->manifold_count <= PL_IMPULSE_MAX_MANIFOLDS);
    const int manifold_count = rigid->manifold_count;
    int robot_contact = 0;
    for (int index = 0; index < manifold_count; ++index) {
        int body = rigid->manifolds[index].body_b;
        robot_contact |= body >= RA_CUDA_BODY_SHELL_START && body < RA_CUDA_ROBOT_BODY_END;
    }
    if (!robot_contact) {
        return;
    }
    RaGripperCollisionFrame frame = ra_gripf(links, state->end_effector);
    RaVec3 hand_axis = ra_rotate(frame.hand.rotation, ra_v3(0, 1, 0));
    for (int manifold_index = 0; manifold_index < manifold_count; ++manifold_index) {
        PlImpulseManifold* manifold = &rigid->manifolds[manifold_index];
        int body = manifold->body_b;
        int robot_body = body >= RA_CUDA_BODY_SHELL_START && body < RA_CUDA_ROBOT_BODY_END;
        int last_joint = body >= RA_CUDA_BODY_LINK_START && body < RA_CUDA_BODY_PAD_LEFT_START
            ? body - RA_CUDA_BODY_LINK_START
            : RA_DOF - 1;
        int side = ra_pad_side(body);
        int group_points[PL_IMPULSE_MAX_POINTS] = {1, 1, 1, 1};
        for (int point = 0; side >= 0 && point < manifold->point_count; ++point) {
            for (int member = point + 1; member < manifold->point_count; ++member) {
                if (manifold->points[member].patch_group == manifold->points[point].patch_group) {
                    ++group_points[point];
                    ++group_points[member];
                }
            }
        }
        for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            memset(&point->reaction, 0, sizeof(point->reaction));
            if (!robot_body) {
                continue;
            }
            PlImpulseReaction* reaction = &point->reaction;
            RaVec3 directions[3] = {manifold->normal, manifold->tangent_1, manifold->tangent_2};
            for (int direction_index = 0; direction_index < 3; ++direction_index) {
                float inverse_mass = 0.0f;
                RaDynamics::response(mass_factor, origins, axes, last_joint, point->point_b,
                    directions[direction_index], ra_v3(0, 0, 0),
                    reaction->robot_jacobian[direction_index],
                    reaction->robot_response[direction_index], &inverse_mass);
                reaction->inverse_mass[direction_index] = inverse_mass;
            }
            if (side >= 0) {
                RaVec3 outward = ra_scale(hand_axis, side == 0 ? 1.0f : -1.0f);
                for (int direction_index = 0; direction_index < 3; ++direction_index) {
                    float jaw_jacobian = 0.5f * ra_dot(outward, directions[direction_index]);
                    reaction->jaw_jacobian[direction_index] = jaw_jacobian;
                    reaction->inverse_mass[direction_index] +=
                        jaw_jacobian * jaw_jacobian / RA_GRIPPER_EFFECTIVE_MASS;
                    reaction->jaw_velocity_response[direction_index] =
                        -jaw_jacobian / RA_GRIPPER_EFFECTIVE_MASS;
                }
            }
            reaction->active = 1;
            if (side >= 0 && point->patch_group != 0 && point->patch.area > 0.0f) {
                float area = point->patch.area / (float)ra_max(group_points[point_index], 1);
                float stiffness =
                    RA_PAD_ELASTIC_MODULUS * area / ra_max(RA_PAD_LAYER_THICKNESS, 1.0e-8f);
                float inverse_effective_mass =
                    RaImpulse::effective_mass(&rigid->bodies[manifold->body_a], point->point_a,
                        &rigid->bodies[manifold->body_b], point->point_b, manifold->normal,
                        reaction, 0);
                float effective_mass = 1.0f / ra_max(inverse_effective_mass, PL_IMPULSE_EPSILON);
                auto compliance = RaExplicit::compliance(stiffness, effective_mass,
                    RA_PAD_DAMPING_RATIO, RA_PHYSICS_DT, PL_IMPULSE_EPSILON);
                point->normal_cfm = compliance.cfm;
                point->normal_erp = compliance.erp;
            }
        }
        memset(&manifold->angular_reaction, 0, sizeof(manifold->angular_reaction));
        if (side >= 0 && manifold->point_count > 0 && manifold->torsional_radius > 0.0f) {
            PlImpulsePoint* point = &manifold->points[0];
            RaDynamics::response(mass_factor, origins, axes, RA_DOF - 1, point->point_b,
                ra_v3(0, 0, 0), manifold->normal, manifold->angular_reaction.robot_jacobian,
                manifold->angular_reaction.robot_response,
                &manifold->angular_reaction.inverse_mass);
            manifold->angular_reaction.active = 1;
        }
    }
}

RA_D static RA_INLINE int ra_pad_contact(
    RaCudaProductionWorld* production, int side, int object_body) {
    RaCudaRigidWorld* world = &production->rigid;
    if (!ra_manok(world)) {
        return 0;
    }
    int start = side == 0 ? RA_CUDA_BODY_PAD_LEFT_START : RA_CUDA_BODY_PAD_RIGHT_START;
    RaVec3 inward = ra_scale(ra_rotate(world->shapes[start].pose.rotation, ra_v3(0, 1, 0)), -1.0f);
    RaCompound::Options options;
    options.dt = RA_PHYSICS_DT;
    options.margin = RA_CONTACT_MARGIN;
    options.static_friction = RA_FINGER_FRICTION;
    options.dynamic_friction = RA_FINGER_FRICTION;
    options.boundary_epsilon = RA_PAD_CSG_BOUNDARY_EPSILON;
    options.support_tolerance = RA_PAD_SUPPORT_PLANE_TOLERANCE;
    options.feature_namespace = (uint32_t)side;
    options.group_base = (uint32_t)side << 8;
    PlImpulseManifold* manifold = &world->manifolds[world->manifold_count];
    RaCompound::Result result;
    if (world->shapes[object_body].type == RA_CONVEX_SPHERE) {
        result = RaCompound::sphere(world->shapes[object_body], world->bodies[object_body],
            world->shapes + start, world->bodies + start, RA_PAD_BOXES, inward, options,
            object_body, start, manifold);
    } else {
        result = RaCompound::box(world->shapes[object_body], world->bodies[object_body],
            world->shapes + start, world->bodies + start, RA_PAD_BOXES, inward, options,
            world->compound_candidate_scratch, object_body, start, manifold);
    }
    assert(result.status != RaCompound::invalid && result.status != RaCompound::capacity);
    if (result.status != RaCompound::contact) {
        return 0;
    }
    if (world->shapes[object_body].type == RA_CONVEX_SPHERE) {
        world->compound_pad_component_mask[side] = result.component_mask;
    } else {
        world->compound_pad_component_mask[side] |= result.component_mask;
    }
    ++world->manifold_count;
    return 1;
}

RA_D static RA_INLINE void ra_advbox(
    RaCudaProductionWorld* world, int body_index, int other_index) {
    RaCudaRigidWorld* rigid = &world->rigid;
    RaRigidBody* body = &rigid->bodies[body_index];
    const RaRigidBody* other = &rigid->bodies[other_index];
    RaConvexSweep sweep = RaSweep::shape(&rigid->shapes[body_index], &rigid->shapes[other_index],
        body->linear_velocity, body->angular_velocity, other->linear_velocity,
        other->angular_velocity, RA_PHYSICS_DT, 0.0f);
    float normal_speed =
        ra_dot(ra_sub(body->linear_velocity, other->linear_velocity), sweep.contact.normal);
    float advance = sweep.hit && normal_speed < 0.0f ? sweep.toi : RA_PHYSICS_DT;
    body->pose.position = ra_add(body->pose.position, ra_scale(body->linear_velocity, advance));
    body->pose.rotation = ra_qint(body->pose.rotation, body->angular_velocity, advance);
}

RA_D static RA_INLINE RaConvexSweep ra_rimccd(const RaRigidBody* ball, float maximum_time) {
    return RaSweep::sweep_sphere_ring(
        ball->pose.position, ball->linear_velocity, RA_BALL_RADIUS, ra_ring(), maximum_time);
}

RA_D static RA_INLINE int ra_tbllo(RaPose pose, RaVec3 half_extent, float margin) {
    RaVec3 axis_x = ra_rotate(pose.rotation, ra_v3(1, 0, 0));
    RaVec3 axis_y = ra_rotate(pose.rotation, ra_v3(0, 1, 0));
    RaVec3 axis_z = ra_rotate(pose.rotation, ra_v3(0, 0, 1));
    float radius_x = fabsf(axis_x.x) * half_extent.x + fabsf(axis_y.x) * half_extent.y
        + fabsf(axis_z.x) * half_extent.z;
    float radius_y = fabsf(axis_x.y) * half_extent.x + fabsf(axis_y.y) * half_extent.y
        + fabsf(axis_z.y) * half_extent.z;
    float radius_z = fabsf(axis_x.z) * half_extent.x + fabsf(axis_y.z) * half_extent.y
        + fabsf(axis_z.z) * half_extent.z;
    float table_min_x = RA_TABLE_CENTER_X - 0.5f * RA_TABLE_SIZE_X;
    float table_max_x = RA_TABLE_CENTER_X + 0.5f * RA_TABLE_SIZE_X;
    float table_min_z = -0.5f * RA_TABLE_SIZE_Z;
    float table_max_z = 0.5f * RA_TABLE_SIZE_Z;
    return pose.position.x + radius_x >= table_min_x - margin
        && pose.position.x - radius_x <= table_max_x + margin
        && pose.position.z + radius_z >= table_min_z - margin
        && pose.position.z - radius_z <= table_max_z + margin
        && pose.position.y - radius_y < RA_TABLE_TOP + margin;
}

RA_D static RA_INLINE int ra_tblpen(const RaState* state) {
    RaPose links[RA_LINKS];
    RaVec3 end_effector;
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, &end_effector);
    float clearance[RA_DOF + 3];
    ra_table_clearances(links, end_effector, clearance);
    // The table covers the arm's entire reachable workspace. Its top is the
    // forbidden half-space boundary, including poses below the thin slab.
    for (int body = 0; body < RA_DOF + 3; ++body) {
        if (clearance[body] < RA_CONTACT_MARGIN) {
            return 1;
        }
    }
    return 0;
}

RA_D static RA_INLINE void ra_prep(
    RaCudaProductionWorld* world, float mass_factor[RA_DOF][RA_DOF]) {
    RaState* state = &world->state;
    // Keep shared FK outputs local to preserve CUDA floating-point evaluation.
    RaPose links[RA_LINKS];
    RaVec3 origins[RA_DOF], axes[RA_DOF];
    float target_width = world->staged.target_width;
    float motor = RaExplicit::servo(target_width, state->gripper_width, state->gripper_velocity,
        RA_GRIPPER_FORCE_STIFFNESS, RA_GRIPPER_FORCE_DAMPING, RA_GRIPPER_MAX_FORCE);
    state->gripper_velocity += motor / RA_GRIPPER_EFFECTIVE_MASS * RA_PHYSICS_DT;
    float matrix[RA_DOF][RA_DOF];
    float gravity[RA_DOF];
    float rhs[RA_DOF];
    float acceleration[RA_DOF] = {0};
    RaPose dynamic_bodies[RA_DYN_BODIES];
    float armature[RA_DOF];
    for (int joint = 0; joint < RA_DOF; ++joint) {
        armature[joint] = 0.1f;
    }
    ra_fk(state->q, state->gripper_width, links, origins, axes, &state->end_effector);
    ra_dposes(links, dynamic_bodies);
    RaDynamics::mass<RaDynamicsModel>(
        dynamic_bodies, origins, axes, RA_DYN_BODIES, armature, matrix);
    RaDynamics::gravity<RaDynamicsModel>(
        dynamic_bodies, origins, axes, RA_DYN_BODIES, ra_v3(0, -9.81f, 0), gravity);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float kp = joint < 2 ? 4500.0f : (joint < 4 ? 3500.0f : 2000.0f);
        float kd = joint < 2 ? 450.0f : (joint < 4 ? 350.0f : 200.0f);
        float arm_motor = RaExplicit::servo(state->target_q[joint], state->q[joint],
            state->qd[joint], kp, kd, joint < 4 ? 87.0f : 12.0f);
        rhs[joint] = arm_motor - state->qd[joint] + gravity[joint];
        world->staged.energy += fabsf(arm_motor * state->qd[joint]) * RA_PHYSICS_DT;
    }
    RaDynamics::factor(matrix, mass_factor);
    RaDynamics::solve(mass_factor, rhs, acceleration);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        state->qd[joint] =
            RaExplicit::velocity(state->qd[joint], acceleration[joint], RA_PHYSICS_DT, 12.0f);
    }
    if (state->basketball_mode) {
        state->cube_velocity = ra_bvel(state->cube_velocity, RA_PHYSICS_DT);
    } else {
        state->cube_velocity.y -= 9.81f * RA_PHYSICS_DT;
    }
    if (state->stack_mode) {
        state->base_cube_velocity.y -= 9.81f * RA_PHYSICS_DT;
    }
    for (int link = 0; link < RA_LINKS; ++link) {
        world->staged.links[link] = links[link];
    }
    for (int joint = 0; joint < RA_DOF; ++joint) {
        world->staged.origins[joint] = origins[joint];
        world->staged.axes[joint] = axes[joint];
    }
    ra_bodies(world);
}

RA_D static RA_INLINE void ra_solve(RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    RaPose* links = world->staged.links;
    float backboard_incoming_speed = 0.0f;
    RaVec3 backboard_normal = ra_v3(0, 0, 0);
    if (state->basketball_mode) {
        for (int index = 0; index < world->rigid.manifold_count; ++index) {
            const PlImpulseManifold* manifold = &world->rigid.manifolds[index];
            if (manifold->body_a != RA_CUDA_BODY_CUBE || manifold->body_b != RA_CUDA_BODY_BACKBOARD
                || manifold->point_count <= 0) {
                continue;
            }
            float incoming =
                -ra_dot(world->rigid.bodies[RA_CUDA_BODY_CUBE].linear_velocity, manifold->normal);
            if (incoming > backboard_incoming_speed) {
                backboard_incoming_speed = incoming;
                backboard_normal = manifold->normal;
            }
        }
    }
    RaImpulse::sort(world->rigid.manifolds, world->rigid.manifold_count);
    RaImpulse::solve(world->rigid.bodies, world->rigid.body_count, world->rigid.manifolds,
        world->rigid.manifold_count, RA_PHYSICS_DT, &world->rigid.config, &world->rigid.cache,
        state);
    if (backboard_incoming_speed > 0.0f) {
        RaRigidBody* ball = &world->rigid.bodies[RA_CUDA_BODY_CUBE];
        float outgoing_speed = ra_dot(ball->linear_velocity, backboard_normal);
        float rebound_floor = RA_BACKBOARD_RESTITUTION * backboard_incoming_speed;
        if (outgoing_speed < rebound_floor) {
            ball->linear_velocity = ra_add(
                ball->linear_velocity, ra_scale(backboard_normal, rebound_floor - outgoing_speed));
        }
    }
    if (state->basketball_mode) {
        RaCudaRigidWorld* rigid = &world->rigid;
        RaRigidBody* ball = &rigid->bodies[RA_CUDA_BODY_CUBE];
        RaVec3 velocity = ball->linear_velocity;
        float advance = RA_PHYSICS_DT;
        advance = RaSweep::approaching_time(
            RaSweep::shape(&rigid->shapes[RA_CUDA_BODY_CUBE], &rigid->shapes[RA_CUDA_BODY_TABLE],
                velocity, ball->angular_velocity, ra_v3(0, 0, 0), ra_v3(0, 0, 0), RA_PHYSICS_DT,
                0.0f),
            velocity, advance);
        advance = RaSweep::approaching_time(
            RaSweep::shape(&rigid->shapes[RA_CUDA_BODY_CUBE],
                &rigid->shapes[RA_CUDA_BODY_BACKBOARD], velocity, ball->angular_velocity,
                ra_v3(0, 0, 0), ra_v3(0, 0, 0), RA_PHYSICS_DT, 0.0f),
            velocity, advance);
        advance = RaSweep::approaching_time(ra_rimccd(ball, RA_PHYSICS_DT), velocity, advance);
        ball->pose.position = ra_add(ball->pose.position, ra_scale(velocity, advance));
        ball->pose.rotation = ra_qint(ball->pose.rotation, ball->angular_velocity, advance);
    } else {
        ra_advbox(world, RA_CUDA_BODY_CUBE, RA_CUDA_BODY_TABLE);
    }
    if (state->stack_mode) {
        ra_advbox(world, RA_CUDA_BODY_BASE, RA_CUDA_BODY_TABLE);
    }
    float previous_q[RA_DOF];
    float candidate_q[RA_DOF];
    for (int joint = 0; joint < RA_DOF; ++joint) {
        previous_q[joint] = state->q[joint];
        state->q[joint] = RaExplicit::coordinate(state->q[joint], state->qd[joint], RA_PHYSICS_DT,
            ra_jmin(joint), ra_jmax(joint), 0.15f);
        candidate_q[joint] = state->q[joint];
    }
    if (ra_tblpen(state)) {
        state->table_blocked_substeps += 1;
        float valid = 0.0f;
        float invalid = 1.0f;
        for (int iteration = 0; iteration < 8; ++iteration) {
            float fraction = 0.5f * (valid + invalid);
            for (int joint = 0; joint < RA_DOF; ++joint) {
                state->q[joint] =
                    previous_q[joint] + fraction * (candidate_q[joint] - previous_q[joint]);
            }
            if (ra_tblpen(state)) {
                invalid = fraction;
            } else {
                valid = fraction;
            }
        }
        for (int joint = 0; joint < RA_DOF; ++joint) {
            state->q[joint] = previous_q[joint] + valid * (candidate_q[joint] - previous_q[joint]);
            state->qd[joint] *= 0.15f;
        }
    }
    state->gripper_width = RaExplicit::slide(
        state->gripper_width, state->gripper_velocity, RA_PHYSICS_DT, 0.004f, 0.080f);
    const RaRigidBody* cube = &world->rigid.bodies[RA_CUDA_BODY_CUBE];
    state->cube_position = cube->pose.position;
    state->cube_rotation = cube->pose.rotation;
    state->cube_velocity = cube->linear_velocity;
    state->cube_angular_velocity = cube->angular_velocity;
    if (state->stack_mode) {
        const RaRigidBody* base = &world->rigid.bodies[RA_CUDA_BODY_BASE];
        state->base_cube_position = base->pose.position;
        state->base_cube_rotation = base->pose.rotation;
        state->base_cube_velocity = base->linear_velocity;
        state->base_cube_angular_velocity = base->angular_velocity;
    }
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, &state->end_effector);
    int active_pad[2] = {0, 0};
    state->pad_normal_impulse[0] = 0.0f;
    state->pad_normal_impulse[1] = 0.0f;
    state->wrist_linear_impulse = ra_v3(0, 0, 0);
    state->wrist_angular_impulse = ra_v3(0, 0, 0);
    for (int manifold_index = 0; manifold_index < world->rigid.manifold_count; ++manifold_index) {
        PlImpulseManifold* manifold = &world->rigid.manifolds[manifold_index];
        int robot_contact = manifold->body_b >= RA_CUDA_BODY_SHELL_START
            && manifold->body_b < RA_CUDA_ROBOT_BODY_END;
        int side = ra_pad_side(manifold->body_b);
        for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
            PlImpulsePoint* point = &manifold->points[point_index];
            float impulse = ra_max(point->normal_impulse, 0.0f);
            if (side >= 0) {
                state->pad_normal_impulse[side] += impulse;
                if (world->rigid.compound_pad_component_mask[side] != 0
                    && point->normal_impulse > 1.0e-8f
                    && point->separation <= RA_CONTACT_MARGIN + 1.0e-6f) {
                    active_pad[side] = 1;
                }
            }
            if (robot_contact && impulse > 0.0f) {
                RaVec3 reaction = ra_scale(manifold->normal, -impulse);
                state->wrist_linear_impulse = ra_add(state->wrist_linear_impulse, reaction);
                state->wrist_angular_impulse = ra_add(state->wrist_angular_impulse,
                    ra_cross(ra_sub(point->point_b, state->end_effector), reaction));
            }
        }
    }
    state->gripper_force =
        0.5f * (state->pad_normal_impulse[0] + state->pad_normal_impulse[1]) / RA_PHYSICS_DT;
    // Grasp flags are task state; pad contacts apply no additional constraint.
    float grip_action = ra_clamp(world->staged.actions[RA_DOF], -1.0f, 1.0f);
    int between_pads = 0;
    if (active_pad[0] && active_pad[1]) {
        RaGripperCollisionFrame frame = ra_gripf(links, state->end_effector);
        RaVec3 left_inward = ra_scale(ra_rotate(frame.left_finger.rotation, ra_v3(0, 1, 0)), -1.0f);
        RaVec3 right_inward =
            ra_scale(ra_rotate(frame.right_finger.rotation, ra_v3(0, 1, 0)), -1.0f);
        between_pads =
            ra_dot(ra_sub(state->cube_position, frame.left_finger.position), left_inward) > 0.0f
            && ra_dot(ra_sub(state->cube_position, frame.right_finger.position), right_inward)
                > 0.0f;
    }
    int pad_pinch = between_pads && state->pad_normal_impulse[0] > 1.0e-7f
        && state->pad_normal_impulse[1] > 1.0e-7f && grip_action < 0.25f;
    int grasp_loss_substeps = state->basketball_mode && grip_action <= 0.25f
        ? RA_BASKETBALL_GRASP_LOSS_SUBSTEPS
        : RA_GRASP_LOSS_SUBSTEPS;
    if (pad_pinch) {
        state->grasp_contact_misses = 0;
        state->episode_pinch_force += state->gripper_force;
        state->pinch_substeps += 1;
    }
    if (!state->grasped && !state->basketball_in_flight && !world->staged.grasp_broken
        && state->grasp_cooldown == 0 && pad_pinch) {
        world->staged.first_grasp |= !state->ever_grasped;
        state->grasped = 1;
        state->ever_grasped = 1;
    } else if (state->grasped && !pad_pinch
        && ++state->grasp_contact_misses >= grasp_loss_substeps) {
        state->grasped = 0;
        world->staged.grasp_broken = 1;
        if (grip_action > 0.25f) {
            state->grasp_cooldown = RA_GRASP_COOLDOWN_STEPS;
            world->staged.released = 1;
        } else {
            state->grasp_cooldown = 0;
            state->slip_events += 1;
        }
    }
    if (state->stack_mode) {
        state->target_position = ra_add(state->base_cube_position,
            ra_v3(0, ra_csup(state->base_cube_rotation, ra_v3(0, 1, 0)) + RA_CUBE_HALF, 0));
    }
}

RA_D static RA_INLINE void ra_objr(RaCudaProductionWorld* world, int object_body, int pad_mask) {
    RaCudaRigidWorld* rigid = &world->rigid;
    for (int item = 0; item < RA_CUDA_SHELL_BOXES; ++item) {
        if ((item < 3 && world->state.basketball_mode) || (item == 3 && (pad_mask & 1))
            || (item == 4 && (pad_mask & 2))) {
            continue;
        }
        ra_pair(rigid, object_body, RA_CUDA_BODY_SHELL_START + item, RA_CONTACT_MARGIN,
            RA_HAND_COLLISION_FRICTION, RA_HAND_COLLISION_FRICTION, 0.0f);
    }
    const RaConvexShape* object = &rigid->shapes[object_body];
    float object_radius = ra_brad(object) + RA_CONTACT_MARGIN;
    for (int item = 0; item < RA_DOF; ++item) {
        int link_body = RA_CUDA_BODY_LINK_START + item;
        RaVec3 delta = ra_sub(object->pose.position, rigid->shapes[link_body].pose.position);
        float limit = object_radius + ra_brad(&rigid->shapes[link_body]);
        if (ra_dot(delta, delta) > limit * limit) {
            continue;
        }
        ra_pair(rigid, object_body, link_body, RA_CONTACT_MARGIN, RA_HAND_COLLISION_FRICTION,
            RA_HAND_COLLISION_FRICTION, 0.0f);
    }
}

RA_D static RA_INLINE void ra_buildc(RaCudaProductionWorld* world) {
    RaState* state = &world->state;
    RaCudaRigidWorld* rigid = &world->rigid;
    const float friction = state->stack_mode
        ? RA_STACK_STATIC_FRICTION
        : (state->basketball_mode ? RA_BALL_FRICTION : RA_CUBE_FRICTION);
    const float dynamic_friction = state->stack_mode
        ? RA_STACK_DYNAMIC_FRICTION
        : (state->basketball_mode ? RA_BALL_FRICTION : RA_CUBE_FRICTION);
    const float restitution = state->basketball_mode ? RA_BALL_RESTITUTION : RA_CUBE_RESTITUTION;
    ra_pair(rigid, RA_CUDA_BODY_CUBE, RA_CUDA_BODY_TABLE, RA_CONTACT_MARGIN, friction,
        dynamic_friction, restitution);
    if (state->stack_mode) {
        ra_pair(rigid, RA_CUDA_BODY_BASE, RA_CUDA_BODY_TABLE, RA_CONTACT_MARGIN,
            RA_STACK_STATIC_FRICTION, RA_STACK_DYNAMIC_FRICTION, RA_CUBE_RESTITUTION);
        ra_pair(rigid, RA_CUDA_BODY_CUBE, RA_CUDA_BODY_BASE, RA_CONTACT_MARGIN,
            RA_STACK_STATIC_FRICTION, RA_STACK_DYNAMIC_FRICTION, RA_CUBE_RESTITUTION);
    }
    if (state->basketball_mode) {
        ra_pair(rigid, RA_CUDA_BODY_CUBE, RA_CUDA_BODY_BACKBOARD, RA_CONTACT_MARGIN,
            RA_BACKBOARD_STATIC_FRICTION, RA_BACKBOARD_DYNAMIC_FRICTION, RA_BACKBOARD_RESTITUTION);
        RaConvexContact rim_contact =
            ra_rimq(rigid->bodies[RA_CUDA_BODY_CUBE].pose.position, RA_CONTACT_MARGIN);
        if (rim_contact.hit && ra_manok(rigid)) {
            PlImpulseCandidate rim_candidate;
            memset(&rim_candidate, 0, sizeof(rim_candidate));
            rim_candidate.contact = rim_contact;
            rim_candidate.feature = 0x72000000u;
            PlImpulseManifold* rim_manifold = &rigid->manifolds[rigid->manifold_count];
            int rim_count = RaImpulse::manifold(RA_CUDA_BODY_CUBE, RA_CUDA_BODY_RIM, &rim_candidate,
                1, RA_CONTACT_MARGIN, 0.55f, 0.45f, RA_BALL_RESTITUTION, rim_manifold);
            rigid->manifold_count += rim_count > 0;
        }
    }
    ra_pad_contact(world, 0, RA_CUDA_BODY_CUBE);
    ra_pad_contact(world, 1, RA_CUDA_BODY_CUBE);
    if (world->state.stack_mode) {
        ra_pad_contact(world, 0, RA_CUDA_BODY_BASE);
        ra_pad_contact(world, 1, RA_CUDA_BODY_BASE);
    }
    int pad_mask = (rigid->compound_pad_component_mask[0] != 0 ? 1 : 0)
        | (rigid->compound_pad_component_mask[1] != 0 ? 2 : 0);
    ra_objr(world, RA_CUDA_BODY_CUBE, pad_mask);
    if (world->state.stack_mode) {
        ra_objr(world, RA_CUDA_BODY_BASE, pad_mask);
    }
    for (int body = RA_CUDA_BODY_SHELL_START; body < RA_CUDA_ROBOT_BODY_END; ++body) {
        const RaConvexShape* shape = &rigid->shapes[body];
        if (ra_tbllo(shape->pose, shape->half_extents, RA_CONTACT_MARGIN + 0.002f)) {
            ra_pair(rigid, RA_CUDA_BODY_TABLE, body, RA_CONTACT_MARGIN, 0.80f, 0.70f, 0.0f);
        }
    }
}

enum { RA_CUDA_BLOCK_SIZE = 32 };

typedef struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag;
    int boundary_reached;
    unsigned int rng;
    RaCudaProductionWorld world;
} Env;

static_assert(sizeof(RaState) % sizeof(unsigned int) == 0,
    "Robot-arm CUDA state must remain naturally word aligned");

__global__ void ra_kinit(
    Env* envs, obs_t* observations, float* rewards, float* terminals, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float local_observation[OBS_SIZE];
    ra_observe(&envs[index].world.state, local_observation);
    for (int feature = 0; feature < OBS_SIZE; ++feature) {
        observations[index * OBS_SIZE + feature] = __float2bfloat16(local_observation[feature]);
    }
    rewards[index] = 0.0f;
    terminals[index] = 0.0f;
}

__global__ void ra_kbegin(Env* envs, int start, int count, const float* actions) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    int state_index = start + local;
    RaCudaProductionWorld* world = &envs[state_index].world;
    RaState* state = &world->state;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        world->staged.actions[action] = actions[state_index * RA_ACTIONS + action];
    }
    const float action_span[RA_DOF] = {2.30f, 1.45f, 2.30f, 1.20f, 2.30f, 1.50f, 2.20f};
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float action = ra_clamp(world->staged.actions[joint], -1.0f, 1.0f);
        state->target_q[joint] =
            ra_clamp(ra_jhome(joint) + action * action_span[joint], ra_jmin(joint), ra_jmax(joint));
    }
    float grip_action = ra_clamp(world->staged.actions[RA_DOF], -1.0f, 1.0f);
    world->staged.target_width = 0.004f + 0.076f * 0.5f * (grip_action + 1.0f);
    state->table_blocked_substeps = 0;
    world->staged.energy = 0.0f;
    world->staged.first_grasp = 0;
    world->staged.grasp_broken = 0;
    world->staged.released = 0;
    if (state->grasp_cooldown > 0) {
        state->grasp_cooldown -= 1;
    }
    if (state->basketball_mode) {
        state->previous_ball_position = state->cube_position;
    }
    ra_fk(state->q, state->gripper_width, world->staged.links, NULL, NULL, &state->end_effector);
    if (state->basketball_mode) {
        state->basketball_previous_grasp_center =
            ra_gctr(state->end_effector, world->staged.links[RA_DOF + 1].rotation);
    }
    state->step += 1;
}

__global__ void ra_kphys(Env* envs, int start, int count) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    RaCudaProductionWorld* world = &envs[start + local].world;
    for (int substep = 0; substep < RA_SUBSTEPS; ++substep) {
        float mass_factor[RA_DOF][RA_DOF];
        ra_prep(world, mass_factor);
        ra_buildc(world);
        ra_react(world, mass_factor);
        ra_solve(world);
    }
}

__global__ void ra_kfin(
    Env* envs, int start, int count, obs_t* observations, float* rewards, float* terminals) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    int state_index = start + local;
    Env* env = envs + state_index;
    RaCudaProductionWorld* world = &env->world;
    float reward = world->state.basketball_mode
        ? ra_stepb(&world->state, world->staged.actions, world->staged.energy,
            world->staged.first_grasp, world->staged.released, world->staged.links)
        : ra_stept(&world->state, world->staged.actions, world->staged.energy,
            world->staged.first_grasp, world->staged.released, world->staged.links);
    reward -= RA_TABLE_BLOCKED_COST * (float)world->state.table_blocked_substeps / RA_SUBSTEPS;
    world->state.table_blocked_steps += world->state.table_blocked_substeps > 0;
    world->state.episode_energy += world->staged.energy;
    world->state.episode_return += reward;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        world->state.previous_action[action] = ra_clamp(world->staged.actions[action], -1.0f, 1.0f);
    }
    float terminal = world->state.done ? 1.0f : 0.0f;
    if (terminal != 0.0f) {
        ra_logep(&world->state, &env->log);
        unsigned int topology = ra_topo(&world->state);
        ra_reset(&world->state);
        ra_rbrst(&world->rigid, topology);
    } else if (world->state.basketball_reset) {
        ra_rbrst(&world->rigid, 3u);
        world->state.basketball_reset = 0;
    }
    float local_observation[OBS_SIZE];
    ra_observe(&world->state, local_observation);
    for (int feature = 0; feature < OBS_SIZE; ++feature) {
        observations[state_index * OBS_SIZE + feature] =
            __float2bfloat16(local_observation[feature]);
    }
    rewards[state_index] = reward;
    terminals[state_index] = terminal;
}
