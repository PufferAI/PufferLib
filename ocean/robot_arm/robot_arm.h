#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Native bf16 train (pufferl defines from_float + precision_t before including
// this header): store obs as precision_t so env→rollout is a D2D copy. Standalone
// CPU / float builds keep float obs_t.
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif
#include "pufferenv.h"

#define RA_HD __host__ __device__
#define RA_D __device__
#define RA_INLINE __forceinline__

#define RA_DOF 7
#define RA_ACTIONS 8
#define RA_LINKS (RA_DOF + 3)
#define OBS_SIZE 86
#define NUM_ATNS RA_ACTIONS
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1}
#ifndef RA_SUBSTEPS
#define RA_SUBSTEPS 8
#endif
#define RA_MAX_STEPS 600
#define RA_BASKETBALL_MAX_STEPS 3600
#define RA_CONTROL_DT (1.0f / 60.0f)
#define RA_PHYSICS_DT (RA_CONTROL_DT / (float)RA_SUBSTEPS)
#define RA_TABLE_TOP 0.00f
#define RA_TABLE_CENTER_X 0.20f
#define RA_TABLE_SIZE_X 20.0f
#define RA_TABLE_SIZE_Z 20.0f
#define RA_TABLE_THICKNESS 0.07f
#define RA_CUBE_HALF 0.035f
#define RA_CUBE_FRICTION 0.72f
#define RA_CUBE_RESTITUTION 0.12f
#define RA_BALL_RADIUS 0.028f
#define RA_BALL_MASS 0.080f
#define RA_BALL_FRICTION 0.68f
#define RA_BALL_RESTITUTION 0.72f
#define RA_BACKBOARD_STATIC_FRICTION 0.28f
#define RA_BACKBOARD_DYNAMIC_FRICTION 0.20f
#define RA_BACKBOARD_RESTITUTION 0.82f
#define RA_BALL_LINEAR_DRAG 0.08f
#define RA_HOOP_CENTER_X 1.55f
#define RA_HOOP_CENTER_Y 0.70f
#define RA_HOOP_CENTER_Z -0.35f
#define RA_HOOP_INNER_RADIUS 0.056f
#define RA_RIM_TUBE_RADIUS 0.008f
#define RA_RIM_MAJOR_RADIUS (RA_HOOP_INNER_RADIUS + RA_RIM_TUBE_RADIUS)
#define RA_BACKBOARD_CENTER_Z -0.43f
#define RA_BACKBOARD_CENTER_Y (RA_HOOP_CENTER_Y + 0.10f)
#define RA_BACKBOARD_HALF_X 0.16f
#define RA_BACKBOARD_HALF_Y 0.14f
#define RA_BACKBOARD_HALF_Z 0.008f
#define RA_ARM_GEOMETRIC_REACH_BOUND 1.435f
#define RA_OBS_POS_SCALE (1.0f / RA_ARM_GEOMETRIC_REACH_BOUND)
#define RA_OBS_LIN_VEL_SCALE 0.15f
#define RA_OBS_ANG_VEL_SCALE 0.1f
#define RA_OBS_GRIP_VEL_SCALE 2.5f
#define RA_OBS_TABLE_RANGE 0.15f
#define RA_TABLE_BLOCKED_COST 0.02f
#define RA_BASKETBALL_RELEASE_DISTANCE 1.10f
#define RA_BASKETBALL_GRASP_CENTER_OFFSET 0.0121f
#define RA_BASKETBALL_GRIP_WIDTH 0.044f
#define RA_BASKETBALL_OPEN_WIDTH 0.070f
#define RA_BASKETBALL_CLOSE_DISTANCE 0.018f
#define RA_BASKETBALL_REOPEN_DISTANCE 0.028f
#define RA_BASKETBALL_APPROACH_BUDGET 0.13f
#define RA_GRASP_COOLDOWN_STEPS 6
#define RA_GRASP_LOSS_SUBSTEPS ((RA_SUBSTEPS) < 8 ? 8 : (RA_SUBSTEPS))
#define RA_BASKETBALL_GRASP_LOSS_SUBSTEPS (3 * RA_GRASP_LOSS_SUBSTEPS)
#define RA_BASKETBALL_GROUNDED_RESET_STEPS 15
#define RA_BASKETBALL_GROUNDED_HEIGHT_SLOP 0.006f
#define RA_BASKETBALL_GROUNDED_MAX_VERTICAL_SPEED 0.12f
#define RA_BASKETBALL_RELEASE_READY_QUALITY 0.45f
#define RA_BASKETBALL_PREDICTED_MISS_CAP 2.0f
#define RA_PICK_CUBE_MASS 0.10f
#define RA_STACK_CUBE_MASS 1.00f
#define RA_FINGER_FRICTION 0.80f
#define RA_GRIPPER_MAX_FORCE 100.0f
#define RA_GRIPPER_FORCE_STIFFNESS 1500.0f
#define RA_GRIPPER_FORCE_DAMPING 18.57f
#define RA_GRIPPER_EFFECTIVE_MASS 0.0575f
#define RA_PAD_ELASTIC_MODULUS 25000000.0f
#define RA_PAD_LAYER_THICKNESS 0.001f
#define RA_PAD_DAMPING_RATIO 1.0f
#define RA_PAD_SUPPORT_PLANE_TOLERANCE 2.0e-6f
#define RA_PAD_CSG_BOUNDARY_EPSILON 2.0e-7f
#define RA_LIFT_HEIGHT 0.060f
#define RA_CARRY_HEIGHT 0.030f
#define RA_PLACE_RADIUS 0.070f
#define RA_PLACE_CLEARANCE 0.110f
#define RA_PLACE_SETTLE_SPEED 0.20f
#define RA_PLACE_SETTLE_STEPS 6
#define RA_STACK_STATIC_FRICTION 0.95f
#define RA_STACK_DYNAMIC_FRICTION 0.75f
#define RA_STACK_ALIGN_RADIUS 0.030f
#define RA_STACK_TRANSPORT_RADIUS 0.080f
#define RA_STACK_RELEASE_RADIUS 0.035f
#define RA_STACK_RELEASE_CLEARANCE 0.060f
#define RA_STACK_HOVER_CLEARANCE 0.015f
#define RA_STACK_HEIGHT_TOLERANCE 0.006f
#define RA_STACK_SETTLE_SPEED 0.10f
#define RA_STACK_SETTLE_STEPS 15
#define RA_CONTACT_VELOCITY_ITERS 16
#define RA_CONTACT_POSITION_ITERS 16
#define RA_CONTACT_MARGIN 0.0005f
#define RA_RESTITUTION_THRESHOLD 0.20f
#define RA_PLACE_SETTLE_ANGULAR_SPEED 0.50f
#define RA_STACK_SETTLE_ANGULAR_SPEED 0.25f
#define RA_STACK_UPRIGHT_ERROR 0.025f
#define RA_STACK_HORIZONTAL_PROGRESS_REWARD 12.0f
#define RA_STACK_HEIGHT_PROGRESS_REWARD 10.0f
#define RA_STACK_UPRIGHT_PROGRESS_REWARD 3.0f
#define RA_STACK_SLIP_PENALTY 0.35f
#define RA_PICK_REWARD_SCALE 0.10f
#define RA_STACK_REWARD_SCALE 0.05f
#define RA_PAD_BOXES 5
#define RA_GRIPPER_CLEARANCE_MARGIN 0.0020f
#define RA_HAND_COLLISION_FRICTION 0.80f

typedef struct RaVec3 {
    float x, y, z;
} RaVec3;

typedef struct RaQuat {
    float x, y, z, w;
} RaQuat;

typedef struct RaPose {
    RaVec3 position;
    RaQuat rotation;
} RaPose;

struct Log {
    float score;
    float episode_length;
    float table_blocked_rate;
    float success_rate;
    float grasp_rate;
    float lift_rate;
    float transport_rate;
    float release_rate;
    float return_value;
    float reach_distance;
    float place_distance;
    float energy;
    float pinch_force;
    float slip_rate;
    float stack_rate;
    float stable_stack_rate;
    float stack_alignment_rate;
    float valid_stack_contact_rate;
    float clearance_rate;
    float settle_rate;
    float stack_alignment;
    float base_slide_distance;
    float cube_angular_speed;
    float base_angular_speed;
    float orientation_error;
    float basketball_mode;
    float baskets;
    float release_center_miss_cm_sum;
    float release_center_miss_count;
    float n;
};

typedef struct RaState {
    uint32_t rng;
    int step;
    int done;
    int no_timeout;
    int stack_mode;
    int basketball_mode;
    int basketball_in_flight;
    int basketball_grounded_steps;
    int basketball_reset;
    int baskets;
    int attempts;
    int basketball_grasps;
    int basketball_releases;
    int table_blocked_substeps;
    int table_blocked_steps;
    int grasped;
    int grasp_cooldown;
    int grasp_contact_misses;
    int basketball_close_ready;
    int basketball_release_ready;
    int basketball_release_commanded;
    int ever_grasped;
    int lifted;
    int transported;
    int released_near_target;
    int placement_settle_steps;
    int ever_stacked;
    int stack_aligned;
    int stack_opening_credited;
    int valid_release_achieved;
    int valid_stack_contact;
    int cleared_after_release;
    int max_placement_settle_steps;
    int success;
    int pinch_substeps;
    int slip_events;
    float q[RA_DOF];
    float qd[RA_DOF];
    float target_q[RA_DOF];
    float previous_action[RA_ACTIONS];
    float gripper_width;
    float gripper_velocity;
    float gripper_force;
    RaVec3 end_effector;
    RaVec3 cube_position;
    RaVec3 cube_velocity;
    RaQuat cube_rotation;
    RaVec3 cube_angular_velocity;
    RaVec3 previous_ball_position;
    RaVec3 base_cube_position;
    RaVec3 base_cube_velocity;
    RaQuat base_cube_rotation;
    RaVec3 base_cube_angular_velocity;
    RaVec3 base_cube_start_position;
    RaVec3 previous_base_cube_position;
    RaVec3 target_position;
    RaVec3 basketball_best_grasp_center;
    RaVec3 basketball_previous_grasp_center;
    float basketball_approach_reward_left;
    float basketball_best_open_error;
    float basketball_best_close_error;
    float previous_reach_distance;
    float previous_place_distance;
    float previous_lift_height;
    float previous_grip_error;
    float previous_throw_quality;
    float basketball_release_center_miss_cm_sum;
    float previous_stack_horizontal;
    float previous_stack_drop_error;
    float previous_stack_orientation_error;
    float episode_return;
    float episode_energy;
    float episode_pinch_force;
    float pad_normal_impulse[2];
    RaVec3 wrist_linear_impulse;
    RaVec3 wrist_angular_impulse;
} RaState;

RA_HD static RA_INLINE float ra_min(float a, float b) {
    return a < b ? a : b;
}

RA_HD static RA_INLINE float ra_max(float a, float b) {
    return a > b ? a : b;
}

RA_HD static RA_INLINE float ra_clamp(float value, float low, float high) {
    return ra_min(ra_max(value, low), high);
}

RA_HD static RA_INLINE RaVec3 ra_v3(float x, float y, float z) {
    return (RaVec3){x, y, z};
}

RA_HD static RA_INLINE RaVec3 ra_add(RaVec3 a, RaVec3 b) {
    return ra_v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

RA_HD static RA_INLINE RaVec3 ra_sub(RaVec3 a, RaVec3 b) {
    return ra_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

RA_HD static RA_INLINE RaVec3 ra_scale(RaVec3 value, float scale) {
    return ra_v3(value.x * scale, value.y * scale, value.z * scale);
}

RA_HD static RA_INLINE float ra_dot(RaVec3 a, RaVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

RA_HD static RA_INLINE float ra_length(RaVec3 value) {
    return sqrtf(ra_dot(value, value));
}

RA_HD static RA_INLINE RaQuat ra_quat(float x, float y, float z, float w) {
    return (RaQuat){x, y, z, w};
}

RA_HD static RA_INLINE RaQuat ra_qmul(RaQuat a, RaQuat b) {
    return ra_quat(a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

RA_HD static RA_INLINE RaQuat ra_qnorm(RaQuat value) {
    float inverse = 1.0f
        / sqrtf(
            ra_max(value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w,
                1.0e-12f));
    return ra_quat(value.x * inverse, value.y * inverse, value.z * inverse, value.w * inverse);
}

RA_HD static RA_INLINE RaQuat ra_qconj(RaQuat value) {
    return ra_quat(-value.x, -value.y, -value.z, value.w);
}

RA_HD static RA_INLINE RaQuat ra_qaxis(RaVec3 axis, float angle) {
    float half = 0.5f * angle;
    float sine = sinf(half);
    return ra_quat(axis.x * sine, axis.y * sine, axis.z * sine, cosf(half));
}

RA_HD static RA_INLINE RaVec3 ra_cross(RaVec3 a, RaVec3 b) {
    return ra_v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

RA_HD static RA_INLINE RaVec3 ra_rotate(RaQuat q, RaVec3 value) {
    RaVec3 imaginary = ra_v3(q.x, q.y, q.z);
    RaVec3 doubled_cross = ra_scale(ra_cross(imaginary, value), 2.0f);
    return ra_add(value, ra_add(ra_scale(doubled_cross, q.w), ra_cross(imaginary, doubled_cross)));
}

RA_D static RA_INLINE RaQuat ra_qint(RaQuat rotation, RaVec3 angular_velocity, float dt) {
    float speed = ra_length(angular_velocity);
    if (speed < 1.0e-8f) {
        return ra_qnorm(rotation);
    }
    RaQuat increment = ra_qaxis(ra_scale(angular_velocity, 1.0f / speed), speed * dt);
    return ra_qnorm(ra_qmul(increment, rotation));
}

RA_D static RA_INLINE void ra_caxes(RaQuat rotation, RaVec3 axes[3]) {
    axes[0] = ra_rotate(rotation, ra_v3(1, 0, 0));
    axes[1] = ra_rotate(rotation, ra_v3(0, 1, 0));
    axes[2] = ra_rotate(rotation, ra_v3(0, 0, 1));
}

RA_D static RA_INLINE float ra_csup(RaQuat rotation, RaVec3 direction) {
    RaVec3 axes[3];
    ra_caxes(rotation, axes);
    return RA_CUBE_HALF
        * (fabsf(ra_dot(axes[0], direction)) + fabsf(ra_dot(axes[1], direction))
            + fabsf(ra_dot(axes[2], direction)));
}

RA_HD static RA_INLINE RaVec3 ra_cvert(RaVec3 position, RaQuat rotation, int vertex) {
    RaVec3 local = ra_v3((vertex & 1) ? RA_CUBE_HALF : -RA_CUBE_HALF,
        (vertex & 2) ? RA_CUBE_HALF : -RA_CUBE_HALF, (vertex & 4) ? RA_CUBE_HALF : -RA_CUBE_HALF);
    return ra_add(position, ra_rotate(rotation, local));
}

RA_D static RA_INLINE float ra_cup(RaQuat rotation) {
    RaVec3 axes[3];
    ra_caxes(rotation, axes);
    float best = ra_max(fabsf(axes[0].y), ra_max(fabsf(axes[1].y), fabsf(axes[2].y)));
    return 1.0f - best;
}

RA_HD static RA_INLINE float ra_rand(uint32_t* state, float low, float high) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value ? value : 0x9e3779b9u;
    float unit = (*state >> 8) * (1.0f / 16777216.0f);
    return low + (high - low) * unit;
}

#include "robot_arm_parameters.h"

RA_HD static RA_INLINE float ra_jhome(int joint) {
    const float values[RA_DOF] = RA_MODEL_HOME;
    return values[joint];
}

RA_D static RA_INLINE float ra_jmin(int joint) {
    const float values[RA_DOF] = RA_MODEL_LOWER;
    return values[joint];
}

RA_D static RA_INLINE float ra_jmax(int joint) {
    const float values[RA_DOF] = RA_MODEL_UPPER;
    return values[joint];
}

RA_HD static void ra_fk(const float* q, float gripper_width, RaPose* links, RaVec3* joint_origins,
    RaVec3* joint_axes, RaVec3* end_effector) {
    links[0].position = ra_v3(0, 0, 0);
    links[0].rotation = ra_quat(-0.70710678118f, 0, 0, 0.70710678118f);
    RaPose parent = links[0];
    const RaVec3 joint_offsets[RA_DOF] = RA_MODEL_OFFSETS;
    const float half_sqrt = 0.70710678118f;
    const RaQuat joint_statics[RA_DOF] = RA_MODEL_ROTATIONS(RA_MODEL_XYZW, half_sqrt);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        RaQuat joint_static = joint_statics[joint];
        RaVec3 origin = ra_add(parent.position, ra_rotate(parent.rotation, joint_offsets[joint]));
        RaQuat static_rotation = ra_qnorm(ra_qmul(parent.rotation, joint_static));
        RaVec3 axis = ra_rotate(static_rotation, ra_v3(0, 0, 1));
        if (joint_origins != NULL) {
            joint_origins[joint] = origin;
        }
        if (joint_axes != NULL) {
            joint_axes[joint] = axis;
        }
        parent.position = origin;
        parent.rotation = ra_qnorm(
            ra_qmul(parent.rotation, ra_qmul(joint_static, ra_qaxis(ra_v3(0, 0, 1), q[joint]))));
        links[joint + 1] = parent;
    }
    RaPose hand;
    hand.position = ra_add(parent.position, ra_rotate(parent.rotation, ra_v3(0, 0, 0.107f)));
    hand.rotation = ra_qnorm(ra_qmul(parent.rotation, ra_qaxis(ra_v3(0, 0, 1), -0.78539816339f)));
    float half_width = 0.5f * ra_clamp(gripper_width, 0.0f, 0.08f);
    links[RA_DOF + 1].position =
        ra_add(hand.position, ra_rotate(hand.rotation, ra_v3(0, half_width, 0.0584f)));
    links[RA_DOF + 1].rotation = hand.rotation;
    links[RA_DOF + 2].position =
        ra_add(hand.position, ra_rotate(hand.rotation, ra_v3(0, -half_width, 0.0584f)));
    links[RA_DOF + 2].rotation =
        ra_qnorm(ra_qmul(hand.rotation, ra_qaxis(ra_v3(0, 0, 1), 3.14159265359f)));
    if (end_effector != NULL) {
        *end_effector = ra_add(hand.position, ra_rotate(hand.rotation, ra_v3(0, 0, 0.115f)));
    }
}

#define RA_DYN_BODIES 10

typedef struct RaInertia3 {
    float xx, yy, zz, xy, xz, yz;
} RaInertia3;

RA_D static void ra_dposes(const RaPose* links, RaPose* bodies) {
    for (int body = 0; body < RA_DOF; ++body) {
        bodies[body] = links[body + 1];
    }
    bodies[7].rotation = links[RA_DOF].rotation;
    bodies[7].position =
        ra_add(links[RA_DOF].position, ra_rotate(links[RA_DOF].rotation, ra_v3(0, 0, 0.107f)));
    bodies[7].rotation =
        ra_qnorm(ra_qmul(bodies[7].rotation, ra_qaxis(ra_v3(0, 0, 1), -0.78539816339f)));
    bodies[8] = links[RA_DOF + 1];
    bodies[9] = links[RA_DOF + 2];
}

#include "../../src/puffysics/serial_dynamics.cuh"
#include "../../src/puffysics/explicit_dynamics.cuh"
#include "../../src/puffysics/box_clearance.cuh"

// Preserve the production xyzw math while the engine owns the chain algorithms.
struct RaDynamicsTraits {
    using Vec3 = RaVec3;
    RA_HD static RaVec3 add(RaVec3 a, RaVec3 b) { return ra_add(a, b); }
    RA_HD static RaVec3 sub(RaVec3 a, RaVec3 b) { return ra_sub(a, b); }
    RA_HD static RaVec3 cross(RaVec3 a, RaVec3 b) { return ra_cross(a, b); }
    RA_HD static RaVec3 scale(RaVec3 a, float b) { return ra_scale(a, b); }
    RA_HD static float dot(RaVec3 a, RaVec3 b) { return ra_dot(a, b); }
    RA_HD static float max(float a, float b) { return ra_max(a, b); }
    RA_HD static float min(float a, float b) { return ra_min(a, b); }
    RA_HD static float clamp(float v, float lo, float hi) { return ra_clamp(v, lo, hi); }
    RA_HD static RaVec3 v3(float x, float y, float z) { return ra_v3(x, y, z); }
    RA_HD static RaVec3 rotate(RaQuat q, RaVec3 v) { return ra_rotate(q, v); }
    RA_HD static RaQuat conjugate(RaQuat q) { return ra_qconj(q); }
};
using RaDynamics = PfSerialDynamicsT<RA_DOF, RaDynamicsTraits>;
struct RaDynamicsModel {
    RA_D static float mass(int b) {
        const float value[RA_DYN_BODIES] = RA_MODEL_MASSES;
        return value[b];
    }
    RA_D static RaVec3 com(int b) {
        const RaVec3 value[RA_DYN_BODIES] = RA_MODEL_CENTERS;
        return value[b];
    }
    RA_D static int last(int b) { return b < RA_DOF ? b : RA_DOF - 1; }
    RA_D static RaDynamics::Tensor inertia(int b) {
        const RaInertia3 values[RA_DYN_BODIES] = RA_MODEL_INERTIAS;
        RaInertia3 i = values[b];
        return {i.xx, i.yy, i.zz, i.xy, i.xz, i.yz};
    }
};
using RaExplicit = PfExplicitDynamicsT<RaDynamicsTraits>;
using RaClearance = PfBoxClearanceT<RaDynamicsTraits>;

RA_D static RA_INLINE void ra_applyr(
    RaState* state, const float response[RA_DOF], float magnitude) {
    for (int joint = 0; joint < RA_DOF; ++joint) {
        state->qd[joint] += response[joint] * magnitude;
    }
}

enum { RA_CONVEX_BOX = 0, RA_CONVEX_SPHERE = 1 };

typedef struct RaConvexShape {
    int type;
    RaPose pose;
    RaVec3 half_extents;
} RaConvexShape;

typedef struct RaRigidBody {
    RaPose pose;
    RaVec3 linear_velocity;
    RaVec3 angular_velocity;
    float mass;
    RaInertia3 local_inertia;
} RaRigidBody;

typedef struct RaConvexContact {
    int hit;
    int iterations;
    float separation;
    RaVec3 normal;
    RaVec3 point_a;
    RaVec3 point_b;
} RaConvexContact;

typedef struct RaConvexSweep {
    int hit;
    int iterations;
    float toi;
    RaConvexContact contact;
} RaConvexSweep;

RA_D static RA_INLINE RaVec3 ra_invi(const RaRigidBody* body, RaVec3 vector) {
    return RaExplicit::inverse_inertia(body->local_inertia, body->pose.rotation, vector, 1.0e-18f);
}

RA_D static RA_INLINE void ra_impa(RaRigidBody* body, RaVec3 point, RaVec3 impulse) {
    if (body->mass <= 0.0f) {
        return;
    }
    body->linear_velocity = ra_add(body->linear_velocity, ra_scale(impulse, 1.0f / body->mass));
    body->angular_velocity = ra_add(body->angular_velocity,
        ra_invi(body, ra_cross(ra_sub(point, body->pose.position), impulse)));
}

RA_D static RA_INLINE float ra_impd(const RaRigidBody* body, RaVec3 point, RaVec3 direction) {
    if (body->mass <= 0.0f) {
        return 0.0f;
    }
    RaVec3 lever = ra_sub(point, body->pose.position);
    RaVec3 angular = ra_cross(lever, direction);
    return 1.0f / body->mass + ra_dot(angular, ra_invi(body, angular));
}

RA_D static RA_INLINE float ra_brad(const RaConvexShape* shape) {
    return shape->type == RA_CONVEX_SPHERE ? shape->half_extents.x : ra_length(shape->half_extents);
}

RA_D static RA_INLINE RaConvexShape ra_padsh(RaPose finger, int index) {
    const RaVec3 positions[RA_PAD_BOXES] = {{0, 0.0055f, 0.0445f}, {0.0055f, 0.0020f, 0.0500f},
        {-0.0055f, 0.0020f, 0.0500f}, {0.0055f, 0.0020f, 0.0395f}, {-0.0055f, 0.0020f, 0.0395f}};
    const RaVec3 extents[RA_PAD_BOXES] = {{0.0085f, 0.0040f, 0.0085f}, {0.0030f, 0.0020f, 0.0030f},
        {0.0030f, 0.0020f, 0.0030f}, {0.0030f, 0.0020f, 0.0035f}, {0.0030f, 0.0020f, 0.0035f}};
    assert(index >= 0 && index < RA_PAD_BOXES);
    RaVec3 local_position = positions[index], half_extents = extents[index];
    return (RaConvexShape){
        .type = RA_CONVEX_BOX,
        .pose = {ra_add(finger.position, ra_rotate(finger.rotation, local_position)),
            finger.rotation},
        .half_extents = half_extents,
    };
}

RA_D static RA_INLINE int ra_pad_clearance(
    RaVec3 position, RaQuat rotation, RaPose finger, float margin) {
    RaVec3 inward = ra_scale(ra_rotate(finger.rotation, ra_v3(0, 1, 0)), -1.0f);
    RaVec3 object_axes[3], pad_axes[3];
    ra_caxes(rotation, object_axes);
    ra_caxes(finger.rotation, pad_axes);
    for (int index = 0; index < RA_PAD_BOXES; ++index) {
        RaConvexShape pad = ra_padsh(finger, index);
        if (RaClearance::face(position, object_axes,
                ra_v3(RA_CUBE_HALF, RA_CUBE_HALF, RA_CUBE_HALF), pad.pose.position, pad_axes,
                pad.half_extents, inward, margin)) {
            return 1;
        }
    }
    return 0;
}

typedef struct RaGripperCollisionFrame {
    RaPose hand;
    RaPose left_finger;
    RaPose right_finger;
} RaGripperCollisionFrame;

typedef struct RaCollisionBox {
    RaPose pose;
    RaVec3 half_extent;
} RaCollisionBox;

RA_D static RA_INLINE RaGripperCollisionFrame ra_gripf(const RaPose* links, RaVec3 end_effector) {
    RaGripperCollisionFrame frame;
    frame.hand.rotation = links[RA_DOF + 1].rotation;
    frame.hand.position = ra_sub(end_effector, ra_rotate(frame.hand.rotation, ra_v3(0, 0, 0.115f)));
    frame.left_finger = links[RA_DOF + 1];
    frame.right_finger = links[RA_DOF + 2];
    return frame;
}

RA_D static RA_INLINE RaPose ra_offp(RaPose parent, RaVec3 local_position) {
    return (RaPose){
        .position = ra_add(parent.position, ra_rotate(parent.rotation, local_position)),
        .rotation = parent.rotation,
    };
}

RA_D static RA_INLINE RaCollisionBox ra_linkb(const RaPose* links, int index) {
    const RaVec3 center[RA_DOF] = {
        {-0.00001f, -0.03719f, -0.06850f},
        {-0.00001f, -0.06949f, 0.03720f},
        {0.04124f, 0.02803f, -0.03300f},
        {-0.04126f, 0.03450f, 0.02803f},
        {-0.00001f, 0.03747f, -0.10340f},
        {0.04206f, 0.01523f, 0.00613f},
        {0.01864f, 0.01863f, 0.07940f},
    };
    const RaVec3 half_extent[RA_DOF] = {
        {0.05501f, 0.09220f, 0.12350f},
        {0.05502f, 0.12451f, 0.09220f},
        {0.09626f, 0.08303f, 0.08800f},
        {0.09625f, 0.08950f, 0.08303f},
        {0.05500f, 0.09246f, 0.15560f},
        {0.08996f, 0.06643f, 0.05012f},
        {0.06267f, 0.06265f, 0.02740f},
    };
    assert(index >= 0 && index < RA_DOF);
    return {ra_offp(links[index + 1], center[index]), half_extent[index]};
}

RA_D static RA_INLINE RaCollisionBox ra_gripb(const RaGripperCollisionFrame* frame, int index) {
    const RaVec3 positions[5] = {{0, 0, -0.0055f}, {0, 0, 0.0250f}, {0, 0, 0.0505f},
        {0, 0.0144f, 0.0150f}, {0, 0.0080f, 0.0420f}};
    const RaVec3 extents[5] = {{0.0320f, 0.1040f, 0.0205f}, {0.0240f, 0.1020f, 0.0100f},
        {0.0220f, 0.1010f, 0.0155f}, {0.0105f, 0.0120f, 0.0150f}, {0.0095f, 0.0080f, 0.0120f}};
    assert(index >= 0 && index < 7);
    int hand = index < 3;
    int item = hand ? index : 3 + (index == 4 || index == 6);
    RaPose parent = hand ? frame->hand : (index >= 5 ? frame->right_finger : frame->left_finger);
    return {ra_offp(parent, positions[item]), extents[item]};
}

// Ten signed surface distances: seven links, palm, left finger, right finger.
RA_D static void ra_table_clearances(
    const RaPose* links, RaVec3 end_effector, float clearance[RA_DOF + 3]) {
    RaGripperCollisionFrame frame = ra_gripf(links, end_effector);
    for (int group = 0; group < RA_DOF + 3; ++group) {
        clearance[group] = RA_ARM_GEOMETRIC_REACH_BOUND;
        int boxes = group < RA_DOF ? 1 : (group == RA_DOF ? 3 : 1 + RA_PAD_BOXES);
        for (int item = 0; item < boxes; ++item) {
            RaCollisionBox box;
            if (group < RA_DOF) {
                box = ra_linkb(links, group);
            } else if (group == RA_DOF || item == 0) {
                box = ra_gripb(&frame, group == RA_DOF ? item : (group == RA_DOF + 1 ? 3 : 5));
            } else {
                RaPose finger = group == RA_DOF + 1 ? frame.left_finger : frame.right_finger;
                RaConvexShape pad = ra_padsh(finger, item - 1);
                box = {pad.pose, pad.half_extents};
            }
            RaVec3 axes[3];
            ra_caxes(box.pose.rotation, axes);
            clearance[group] = ra_min(clearance[group],
                RaClearance::plane(box.pose.position, axes, box.half_extent,
                    ra_v3(0, 1, 0), RA_TABLE_TOP));
        }
    }
}

RA_HD static RA_INLINE RaVec3 ra_gctr(RaVec3 end_effector, RaQuat hand_rotation) {
    return ra_sub(
        end_effector, ra_rotate(hand_rotation, ra_v3(0, 0, RA_BASKETBALL_GRASP_CENTER_OFFSET)));
}

RA_HD static RA_INLINE RaVec3 ra_hoop(void) {
    return ra_v3(RA_HOOP_CENTER_X, RA_HOOP_CENTER_Y, RA_HOOP_CENTER_Z);
}

RA_HD static RA_INLINE RaVec3 ra_bvel(RaVec3 velocity, float dt) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    float decay = expf(-drag * dt);
    velocity.x *= decay;
    velocity.y = (velocity.y + gravity / drag) * decay - gravity / drag;
    velocity.z *= decay;
    return velocity;
}

RA_HD static RA_INLINE RaVec3 ra_bpos(RaVec3 position, RaVec3 velocity, float time) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    float decay = expf(-drag * time);
    float travel = (1.0f - decay) / drag;
    return ra_v3(position.x + velocity.x * travel,
        position.y + (velocity.y + gravity / drag) * travel - gravity * time / drag,
        position.z + velocity.z * travel);
}

RA_HD static RA_INLINE float ra_blq(RaVec3 position, RaVec3 velocity) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    RaVec3 delta = ra_sub(ra_hoop(), position);
    float horizontal = sqrtf(delta.x * delta.x + delta.z * delta.z);
    float flight_time = ra_clamp(horizontal / 2.20f, 0.45f, 0.75f);
    float travel = (1.0f - expf(-drag * flight_time)) / drag;
    RaVec3 target = ra_v3(delta.x / travel,
        (delta.y + gravity * flight_time / drag) / travel - gravity / drag, delta.z / travel);
    RaVec3 error = ra_sub(velocity, target);
    float error_squared = ra_dot(error, error);
    const float sigma = 1.50f;
    return expf(-0.5f * error_squared / (sigma * sigma));
}

RA_HD static RA_INLINE int ra_bxing(
    RaVec3 position, RaVec3 velocity, RaVec3* crossing, float* crossing_time, RaVec3* apex) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    RaVec3 hoop = ra_hoop();
    float apex_time = velocity.y > 0.0f ? logf(1.0f + drag * velocity.y / gravity) / drag : 0.0f;
    RaVec3 apex_position = ra_bpos(position, velocity, apex_time);
    if (apex != NULL) {
        *apex = apex_position;
    }
    if (position.y < hoop.y && apex_position.y <= hoop.y) {
        return 0;
    }
    float discriminant = velocity.y * velocity.y + 2.0f * gravity * (position.y - hoop.y);
    if (discriminant <= 0.0f) {
        return 0;
    }
    float time = (velocity.y + sqrtf(discriminant)) / gravity;
    time = ra_clamp(time, apex_time + 1.0e-4f, 2.0f);
    for (int iteration = 0; iteration < 3; ++iteration) {
        float decay = expf(-drag * time);
        float predicted_y = position.y + (velocity.y + gravity / drag) * (1.0f - decay) / drag
            - gravity * time / drag;
        float predicted_vy = (velocity.y + gravity / drag) * decay - gravity / drag;
        if (predicted_vy >= -1.0e-4f) {
            return 0;
        }
        time = ra_clamp(time - (predicted_y - hoop.y) / predicted_vy, apex_time + 1.0e-4f, 2.0f);
    }
    RaVec3 predicted = ra_bpos(position, velocity, time);
    float decay = expf(-drag * time);
    float predicted_vy = (velocity.y + gravity / drag) * decay - gravity / drag;
    if (time <= 0.0f || time >= 2.0f || predicted_vy >= 0.0f
        || fabsf(predicted.y - hoop.y) > 0.02f) {
        return 0;
    }
    predicted.y = hoop.y;
    if (crossing != NULL) {
        *crossing = predicted;
    }
    if (crossing_time != NULL) {
        *crossing_time = time;
    }
    return 1;
}

RA_HD static RA_INLINE float ra_btq(RaVec3 position, RaVec3 velocity) {
    RaVec3 crossing;
    if (!ra_bxing(position, velocity, &crossing, NULL, NULL)) {
        return 0.0f;
    }
    RaVec3 hoop = ra_hoop();
    float dx = crossing.x - hoop.x;
    float dz = crossing.z - hoop.z;
    float radial_error_squared = dx * dx + dz * dz;
    const float coarse_sigma = 0.25f;
    const float fine_sigma = 0.055f;
    float coarse = expf(-0.5f * radial_error_squared / (coarse_sigma * coarse_sigma));
    float fine = expf(-0.5f * radial_error_squared / (fine_sigma * fine_sigma));
    return 0.25f * coarse + 0.75f * fine;
}

#include "robot_arm_task.h"
#include "robot_arm_render.h"
