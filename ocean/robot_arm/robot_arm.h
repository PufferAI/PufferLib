#pragma once

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pufferenv.h"
#define float3 ra_raymath_float3
#include "raymath.h"
#undef float3

#define RA_HD __host__ __device__
#define RA_D __device__
#define RA_INLINE __forceinline__

#define RA_DOF 7
#define RA_ACTIONS 8
#define RA_LINKS (RA_DOF + 3)
#define OBS_SIZE 69
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
#define RA_BASKETBALL_RELEASE_DISTANCE 1.10f
#define RA_BASKETBALL_GRASP_CENTER_OFFSET 0.0121f
#define RA_BASKETBALL_GRIP_WIDTH 0.044f
#define RA_BASKETBALL_OPEN_WIDTH 0.070f
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
    RaVec3 value = {x, y, z};
    return value;
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

RA_HD static RA_INLINE RaVec3 ra_lerp(RaVec3 a, RaVec3 b, float t) {
    return ra_add(a, ra_scale(ra_sub(b, a), t));
}

RA_HD static RA_INLINE float ra_dot(RaVec3 a, RaVec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

RA_HD static RA_INLINE float ra_length(RaVec3 value) {
    return sqrtf(ra_dot(value, value));
}

RA_HD static RA_INLINE RaQuat ra_quat(float x, float y, float z, float w) {
    RaQuat value = {x, y, z, w};
    return value;
}

RA_HD static RA_INLINE RaQuat ra_qmul(RaQuat a, RaQuat b) {
    return ra_quat(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

RA_HD static RA_INLINE RaQuat ra_qnorm(RaQuat value) {
    float inverse = 1.0f / sqrtf(ra_max(value.x*value.x
        + value.y*value.y + value.z*value.z + value.w*value.w, 1.0e-12f));
    return ra_quat(value.x*inverse, value.y*inverse,
        value.z*inverse, value.w*inverse);
}

RA_HD static RA_INLINE RaQuat ra_qconj(RaQuat value) {
    return ra_quat(-value.x, -value.y, -value.z, value.w);
}

RA_HD static RA_INLINE RaQuat ra_qaxis(RaVec3 axis, float angle) {
    float half = 0.5f * angle;
    float sine = sinf(half);
    return ra_quat(axis.x*sine, axis.y*sine, axis.z*sine, cosf(half));
}

RA_HD static RA_INLINE RaVec3 ra_cross(RaVec3 a, RaVec3 b) {
    return ra_v3(a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

RA_HD static RA_INLINE RaVec3 ra_rotate(RaQuat q, RaVec3 value) {
    RaVec3 imaginary = ra_v3(q.x, q.y, q.z);
    RaVec3 doubled_cross = ra_scale(ra_cross(imaginary, value), 2.0f);
    return ra_add(value, ra_add(ra_scale(doubled_cross, q.w),
        ra_cross(imaginary, doubled_cross)));
}

RA_D static RA_INLINE RaQuat ra_qint(
        RaQuat rotation, RaVec3 angular_velocity, float dt) {
    float speed = ra_length(angular_velocity);
    if (speed < 1.0e-8f) {
        return ra_qnorm(rotation);
    }
    RaQuat increment = ra_qaxis(
        ra_scale(angular_velocity, 1.0f / speed), speed * dt);
    return ra_qnorm(ra_qmul(increment, rotation));
}

RA_D static RA_INLINE void ra_caxes(RaQuat rotation, RaVec3 axes[3]) {
    axes[0] = ra_rotate(rotation, ra_v3(1, 0, 0));
    axes[1] = ra_rotate(rotation, ra_v3(0, 1, 0));
    axes[2] = ra_rotate(rotation, ra_v3(0, 0, 1));
}

RA_D static RA_INLINE float ra_csup(
        RaQuat rotation, RaVec3 direction) {
    RaVec3 axes[3];
    ra_caxes(rotation, axes);
    return RA_CUBE_HALF * (fabsf(ra_dot(axes[0], direction))
        + fabsf(ra_dot(axes[1], direction))
        + fabsf(ra_dot(axes[2], direction)));
}

RA_HD static RA_INLINE RaVec3 ra_cvert(
        RaVec3 position, RaQuat rotation, int vertex) {
    RaVec3 local = ra_v3(
        (vertex & 1) ? RA_CUBE_HALF : -RA_CUBE_HALF,
        (vertex & 2) ? RA_CUBE_HALF : -RA_CUBE_HALF,
        (vertex & 4) ? RA_CUBE_HALF : -RA_CUBE_HALF);
    return ra_add(position, ra_rotate(rotation, local));
}

RA_D static RA_INLINE float ra_cup(RaQuat rotation) {
    RaVec3 axes[3];
    ra_caxes(rotation, axes);
    float best = ra_max(fabsf(axes[0].y),
        ra_max(fabsf(axes[1].y), fabsf(axes[2].y)));
    return 1.0f - best;
}

RA_HD static RA_INLINE float ra_rand(
        uint32_t* state, float low, float high) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value ? value : 0x9e3779b9u;
    float unit = (*state >> 8) * (1.0f / 16777216.0f);
    return low + (high - low) * unit;
}

RA_HD static RA_INLINE float ra_jhome(int joint) {
    const float values[RA_DOF] = {0.00f, -0.785398f, 0.00f, -2.356194f,
        0.00f, 1.570796f, 0.785398f};
    return values[joint];
}

RA_D static RA_INLINE float ra_jmin(int joint) {
    const float values[RA_DOF] = {-2.8973f, -1.7628f, -2.8973f, -3.0718f,
        -2.8973f, -0.0175f, -2.8973f};
    return values[joint];
}

RA_D static RA_INLINE float ra_jmax(int joint) {
    const float values[RA_DOF] = {2.8973f, 1.7628f, 2.8973f, -0.0698f,
        2.8973f, 3.7525f, 2.8973f};
    return values[joint];
}

RA_D static RA_INLINE float ra_aspan(int joint) {
    const float values[RA_DOF] = {2.30f, 1.45f, 2.30f, 1.20f,
        2.30f, 1.50f, 2.20f};
    return values[joint];
}

RA_HD static void ra_fk(const float* q, float gripper_width, RaPose* links,
        RaVec3* joint_origins, RaVec3* joint_axes, RaVec3* end_effector) {
    links[0].position = ra_v3(0, 0, 0);
    links[0].rotation = ra_quat(-0.70710678118f, 0, 0, 0.70710678118f);
    RaPose parent = links[0];
    const RaVec3 joint_offsets[RA_DOF] = {
        {0, 0, 0.333f}, {0, 0, 0}, {0, -0.316f, 0},
        {0.0825f, 0, 0}, {-0.0825f, 0.384f, 0},
        {0, 0, 0}, {0.088f, 0, 0},
    };
    const float half_sqrt = 0.70710678118f;
    const RaQuat joint_statics[RA_DOF] = {
        {0, 0, 0, 1},
        {-half_sqrt, 0, 0, half_sqrt},
        {half_sqrt, 0, 0, half_sqrt},
        {half_sqrt, 0, 0, half_sqrt},
        {-half_sqrt, 0, 0, half_sqrt},
        {half_sqrt, 0, 0, half_sqrt},
        {half_sqrt, 0, 0, half_sqrt},
    };
    for (int joint = 0; joint < RA_DOF; ++joint) {
        RaQuat joint_static = joint_statics[joint];
        RaVec3 origin = ra_add(parent.position,
            ra_rotate(parent.rotation, joint_offsets[joint]));
        RaQuat static_rotation = ra_qnorm(
            ra_qmul(parent.rotation, joint_static));
        RaVec3 axis = ra_rotate(static_rotation, ra_v3(0, 0, 1));
        if (joint_origins != NULL) {
            joint_origins[joint] = origin;
        }
        if (joint_axes != NULL) {
            joint_axes[joint] = axis;
        }
        parent.position = origin;
        parent.rotation = ra_qnorm(ra_qmul(parent.rotation,
            ra_qmul(joint_static, ra_qaxis(ra_v3(0, 0, 1), q[joint]))));
        links[joint + 1] = parent;
    }
    RaPose hand;
    hand.position = ra_add(parent.position,
        ra_rotate(parent.rotation, ra_v3(0, 0, 0.107f)));
    hand.rotation = ra_qnorm(ra_qmul(parent.rotation,
        ra_qaxis(ra_v3(0, 0, 1), -0.78539816339f)));
    float half_width = 0.5f * ra_clamp(gripper_width, 0.0f, 0.08f);
    links[RA_DOF + 1].position = ra_add(hand.position,
        ra_rotate(hand.rotation, ra_v3(0, half_width, 0.0584f)));
    links[RA_DOF + 1].rotation = hand.rotation;
    links[RA_DOF + 2].position = ra_add(hand.position,
        ra_rotate(hand.rotation, ra_v3(0, -half_width, 0.0584f)));
    links[RA_DOF + 2].rotation = ra_qnorm(ra_qmul(hand.rotation,
        ra_qaxis(ra_v3(0, 0, 1), 3.14159265359f)));
    if (end_effector != NULL) {
        *end_effector = ra_add(hand.position,
            ra_rotate(hand.rotation, ra_v3(0, 0, 0.115f)));
    }
}

#define RA_DYN_BODIES 10

typedef struct RaInertia3 {
    float xx, yy, zz, xy, xz, yz;
} RaInertia3;

RA_D static RA_INLINE float ra_dmass(int body) {
    const float value[RA_DYN_BODIES] = {
        4.970684f, 0.646926f, 3.228604f, 3.587895f, 1.225946f,
        1.666555f, 0.735522f, 0.730000f, 0.015000f, 0.015000f,
    };
    return value[body];
}

RA_D static RA_INLINE RaVec3 ra_dcom(int body) {
    const RaVec3 value[RA_DYN_BODIES] = {
        { 0.003875f,  0.002081f, -0.047620f},
        {-0.003141f, -0.028720f,  0.003495f},
        { 0.027518f,  0.039252f, -0.066502f},
        {-0.053170f,  0.104419f,  0.027454f},
        {-0.011953f,  0.041065f, -0.038437f},
        { 0.060149f, -0.014117f, -0.010517f},
        { 0.010517f, -0.004252f,  0.061597f},
        {-0.010000f,  0.000000f,  0.030000f},
        { 0.000000f,  0.000000f,  0.000000f},
        { 0.000000f,  0.000000f,  0.000000f},
    };
    return value[body];
}

RA_D static RA_INLINE RaInertia3 ra_dinert(int body) {
    const RaInertia3 value[RA_DYN_BODIES] = {
        {0.703370f, 0.706610f, 0.009117f,
            -0.000139f,  0.006772f,  0.019169f},
        {0.007962f, 0.028110f, 0.025995f,
            -0.003925f,  0.010254f,  0.000704f},
        {0.037242f, 0.036155f, 0.010830f,
            -0.004761f, -0.011396f, -0.012805f},
        {0.025853f, 0.019552f, 0.028323f,
             0.007796f, -0.001332f,  0.008641f},
        {0.035549f, 0.029474f, 0.008627f,
            -0.002117f, -0.004037f,  0.000229f},
        {0.001964f, 0.004354f, 0.005433f,
             0.000109f, -0.001158f,  0.000341f},
        {0.012516f, 0.010027f, 0.004815f,
            -0.000428f, -0.001196f, -0.000741f},
        {0.001000f, 0.002500f, 0.001700f,
             0.000000f,  0.000000f,  0.000000f},
        {0.000002375f, 0.000002375f, 0.000000750f,
             0.000000000f, 0.000000000f, 0.000000000f},
        {0.000002375f, 0.000002375f, 0.000000750f,
             0.000000000f, 0.000000000f, 0.000000000f},
    };
    return value[body];
}

RA_D static RA_INLINE int ra_dlast(int body) {
    return body < RA_DOF ? body : RA_DOF - 1;
}

RA_D static void ra_dpose(const float* q, float gripper_width,
        RaPose* bodies, RaVec3* origins, RaVec3* axes) {
    RaPose links[RA_LINKS];
    ra_fk(q, gripper_width, links, origins, axes, NULL);
    for (int body = 0; body < RA_DOF; ++body) {
        bodies[body] = links[body + 1];
    }
    bodies[7].rotation = links[RA_DOF].rotation;
    bodies[7].position = ra_add(links[RA_DOF].position,
        ra_rotate(links[RA_DOF].rotation, ra_v3(0, 0, 0.107f)));
    bodies[7].rotation = ra_qnorm(ra_qmul(bodies[7].rotation,
        ra_qaxis(ra_v3(0, 0, 1), -0.78539816339f)));
    bodies[8] = links[RA_DOF + 1];
    bodies[9] = links[RA_DOF + 2];
}

RA_D static void ra_massm(
        const RaState* state, float matrix[RA_DOF][RA_DOF]) {
    for (int row = 0; row < RA_DOF; ++row) {
        for (int column = 0; column < RA_DOF; ++column) {
            matrix[row][column] = 0.0f;
        }
    }
    RaPose bodies[RA_DYN_BODIES];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
    ra_dpose(state->q, state->gripper_width, bodies, origins, axes);
    for (int body = 0; body < RA_DYN_BODIES; ++body) {
        float mass = ra_dmass(body);
        RaInertia3 inertia = ra_dinert(body);
        RaVec3 com = ra_add(bodies[body].position,
            ra_rotate(bodies[body].rotation, ra_dcom(body)));
        int last = ra_dlast(body);
        for (int row = 0; row <= last; ++row) {
            RaVec3 linear_row = ra_cross(
                axes[row], ra_sub(com, origins[row]));
            RaVec3 local_axis = ra_rotate(
                ra_qconj(bodies[body].rotation), axes[row]);
            RaVec3 local_inertia = ra_v3(
                inertia.xx*local_axis.x + inertia.xy*local_axis.y
                    + inertia.xz*local_axis.z,
                inertia.xy*local_axis.x + inertia.yy*local_axis.y
                    + inertia.yz*local_axis.z,
                inertia.xz*local_axis.x + inertia.yz*local_axis.y
                    + inertia.zz*local_axis.z);
            RaVec3 inertia_row = ra_rotate(
                bodies[body].rotation, local_inertia);
            for (int column = 0; column <= row; ++column) {
                RaVec3 linear_column = ra_cross(
                    axes[column], ra_sub(com, origins[column]));
                float value = mass * ra_dot(linear_row, linear_column)
                    + ra_dot(axes[column], inertia_row);
                matrix[row][column] += value;
                if (row != column) {
                    matrix[column][row] += value;
                }
            }
        }
    }
    for (int joint = 0; joint < RA_DOF; ++joint) {
        matrix[joint][joint] += 0.1f;
    }
}

RA_D static void ra_massf(
        const float matrix[RA_DOF][RA_DOF],
        float lower[RA_DOF][RA_DOF]) {
    for (int row = 0; row < RA_DOF; ++row) {
        for (int column = 0; column <= row; ++column) {
            float sum = matrix[row][column];
            for (int k = 0; k < column; ++k) {
                sum -= lower[row][k] * lower[column][k];
            }
            if (row == column) {
                lower[row][column] = sqrtf(ra_max(sum, 1.0e-8f));
            } else {
                lower[row][column] = sum / lower[column][column];
            }
        }
    }
}

RA_D static void ra_masss(
        const float lower[RA_DOF][RA_DOF],
        const float rhs[RA_DOF], float solution[RA_DOF]) {
    float y[RA_DOF] = {0};
    for (int row = 0; row < RA_DOF; ++row) {
        float sum = rhs[row];
        for (int k = 0; k < row; ++k) {
            sum -= lower[row][k] * y[k];
        }
        y[row] = sum / lower[row][row];
    }
    for (int row = RA_DOF - 1; row >= 0; --row) {
        float sum = y[row];
        for (int k = row + 1; k < RA_DOF; ++k) {
            sum -= lower[k][row] * solution[k];
        }
        solution[row] = sum / lower[row][row];
    }
}

RA_D static void ra_gravt(const RaState* state,
        float torque[RA_DOF]) {
    for (int joint = 0; joint < RA_DOF; ++joint) {
        torque[joint] = 0.0f;
    }
    RaPose bodies[RA_DYN_BODIES];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
    ra_dpose(state->q, state->gripper_width,
        bodies, origins, axes);
    const RaVec3 gravity = {0, -9.81f, 0};
    for (int body = 0; body < RA_DYN_BODIES; ++body) {
        float mass = ra_dmass(body);
        RaVec3 com = ra_add(bodies[body].position,
            ra_rotate(bodies[body].rotation, ra_dcom(body)));
        int last = ra_dlast(body);
        for (int joint = 0; joint <= last; ++joint) {
            RaVec3 linear = ra_cross(
                axes[joint], ra_sub(com, origins[joint]));
            torque[joint] += mass * ra_dot(linear, gravity);
        }
    }
}

RA_D static RA_INLINE float ra_mlim(int joint) {
    return joint < 4 ? 87.0f : 12.0f;
}

RA_D static void ra_jacfk(
        const float lower[RA_DOF][RA_DOF], const RaVec3* origins,
        const RaVec3* axes, int last_joint, RaVec3 point,
        RaVec3 linear_direction, RaVec3 angular_direction,
        float jacobian[RA_DOF], float response[RA_DOF],
        float* inverse_mass) {
    for (int joint = 0; joint < RA_DOF; ++joint) {
        if (joint <= last_joint) {
            RaVec3 linear = ra_cross(
                axes[joint], ra_sub(point, origins[joint]));
            jacobian[joint] = ra_dot(linear, linear_direction)
                + ra_dot(axes[joint], angular_direction);
        } else {
            jacobian[joint] = 0.0f;
        }
        response[joint] = 0.0f;
    }
    ra_masss(lower, jacobian, response);
    *inverse_mass = 0.0f;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        *inverse_mass += jacobian[joint] * response[joint];
    }
}

RA_D static RA_INLINE RaVec3 ra_ptvel(
        const float* qd, const RaVec3* origins, const RaVec3* axes,
        int last_joint, RaVec3 point) {
    RaVec3 velocity = ra_v3(0, 0, 0);
    for (int joint = 0; joint <= last_joint; ++joint) {
        velocity = ra_add(velocity, ra_scale(
            ra_cross(axes[joint], ra_sub(point, origins[joint])),
            qd[joint]));
    }
    return velocity;
}

RA_D static RA_INLINE RaVec3 ra_angvel(
        const float* qd, const RaVec3* axes, int last_joint) {
    RaVec3 velocity = ra_v3(0, 0, 0);
    for (int joint = 0; joint <= last_joint; ++joint) {
        velocity = ra_add(velocity, ra_scale(axes[joint], qd[joint]));
    }
    return velocity;
}

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

RA_D static RA_INLINE RaVec3 ra_invi(
        const RaRigidBody* body, RaVec3 vector) {
    RaInertia3 inertia = body->local_inertia;
    RaVec3 local = ra_rotate(ra_qconj(body->pose.rotation), vector);
    float cofactor_xx = inertia.yy*inertia.zz - inertia.yz*inertia.yz;
    float cofactor_xy = inertia.xz*inertia.yz - inertia.xy*inertia.zz;
    float cofactor_xz = inertia.xy*inertia.yz - inertia.xz*inertia.yy;
    float cofactor_yy = inertia.xx*inertia.zz - inertia.xz*inertia.xz;
    float cofactor_yz = inertia.xy*inertia.xz - inertia.xx*inertia.yz;
    float cofactor_zz = inertia.xx*inertia.yy - inertia.xy*inertia.xy;
    float determinant = inertia.xx*cofactor_xx
        + inertia.xy*cofactor_xy + inertia.xz*cofactor_xz;
    float inverse = 1.0f / ra_max(fabsf(determinant), 1.0e-18f);
    if (determinant < 0.0f) {
        inverse = -inverse;
    }
    RaVec3 product = ra_v3(
        inverse*(cofactor_xx*local.x + cofactor_xy*local.y
            + cofactor_xz*local.z),
        inverse*(cofactor_xy*local.x + cofactor_yy*local.y
            + cofactor_yz*local.z),
        inverse*(cofactor_xz*local.x + cofactor_yz*local.y
            + cofactor_zz*local.z));
    return ra_rotate(body->pose.rotation, product);
}

RA_D static RA_INLINE void ra_impa(
        RaRigidBody* body, RaVec3 point, RaVec3 impulse) {
    if (body->mass <= 0.0f) {
        return;
    }
    body->linear_velocity = ra_add(body->linear_velocity,
        ra_scale(impulse, 1.0f / body->mass));
    body->angular_velocity = ra_add(body->angular_velocity,
        ra_invi(body,
            ra_cross(ra_sub(point, body->pose.position), impulse)));
}

RA_D static RA_INLINE float ra_impd(
        const RaRigidBody* body, RaVec3 point, RaVec3 direction) {
    if (body->mass <= 0.0f) {
        return 0.0f;
    }
    RaVec3 lever = ra_sub(point, body->pose.position);
    RaVec3 angular = ra_cross(lever, direction);
    return 1.0f / body->mass + ra_dot(angular,
        ra_invi(body, angular));
}

RA_D static RA_INLINE float ra_brad(
        const RaConvexShape* shape) {
    return shape->type == RA_CONVEX_SPHERE
        ? shape->half_extents.x : ra_length(shape->half_extents);
}

RA_D static RA_INLINE RaConvexShape ra_padsh(
        RaPose finger, int index) {
    RaVec3 local_position;
    RaVec3 half_extents;
    if (index == 0) {
        local_position = ra_v3(0.0f, 0.0055f, 0.0445f);
        half_extents = ra_v3(0.0085f, 0.0040f, 0.0085f);
    } else if (index == 1) {
        local_position = ra_v3(0.0055f, 0.0020f, 0.0500f);
        half_extents = ra_v3(0.0030f, 0.0020f, 0.0030f);
    } else if (index == 2) {
        local_position = ra_v3(-0.0055f, 0.0020f, 0.0500f);
        half_extents = ra_v3(0.0030f, 0.0020f, 0.0030f);
    } else if (index == 3) {
        local_position = ra_v3(0.0055f, 0.0020f, 0.0395f);
        half_extents = ra_v3(0.0030f, 0.0020f, 0.0035f);
    } else {
        local_position = ra_v3(-0.0055f, 0.0020f, 0.0395f);
        half_extents = ra_v3(0.0030f, 0.0020f, 0.0035f);
    }
    RaConvexShape pad;
    memset(&pad, 0, sizeof(pad));
    pad.type = RA_CONVEX_BOX;
    pad.pose.position = ra_add(finger.position,
        ra_rotate(finger.rotation, local_position));
    pad.pose.rotation = finger.rotation;
    pad.half_extents = half_extents;
    return pad;
}

RA_D static RA_INLINE float ra_oboxr(
        const RaVec3 axes[3], RaVec3 half_extents, RaVec3 direction) {
    return half_extents.x * fabsf(ra_dot(axes[0], direction))
        + half_extents.y * fabsf(ra_dot(axes[1], direction))
        + half_extents.z * fabsf(ra_dot(axes[2], direction));
}

RA_D static RA_INLINE int ra_padhit(
        RaVec3 cube_position, RaQuat cube_rotation,
        RaPose finger, float margin, RaConvexContact* best) {
    RaVec3 inward = ra_scale(
        ra_rotate(finger.rotation, ra_v3(0, 1, 0)), -1.0f);
    RaVec3 cube_axes[3];
    RaVec3 pad_axes[3];
    ra_caxes(cube_rotation, cube_axes);
    ra_caxes(finger.rotation, pad_axes);
    int found = 0;
    RaVec3 point_a_sum = ra_v3(0, 0, 0);
    RaVec3 point_b_sum = ra_v3(0, 0, 0);
    int manifold_points = 0;
    memset(best, 0, sizeof(*best));
    best->separation = 1.0e30f;
    for (int index = 0; index < RA_PAD_BOXES; ++index) {
        RaConvexShape pad = ra_padsh(finger, index);
        RaConvexContact candidate;
        memset(&candidate, 0, sizeof(candidate));
        RaVec3 delta = ra_sub(cube_position, pad.pose.position);
        RaVec3 cube_half = ra_v3(RA_CUBE_HALF, RA_CUBE_HALF, RA_CUBE_HALF);
        int sat_ok = 1;
        for (int axis_index = 0; axis_index < 15; ++axis_index) {
            RaVec3 axis;
            if (axis_index < 3) {
                axis = cube_axes[axis_index];
            } else if (axis_index < 6) {
                axis = pad_axes[axis_index - 3];
            } else {
                int pair = axis_index - 6;
                axis = ra_cross(cube_axes[pair / 3], pad_axes[pair % 3]);
                float length = ra_length(axis);
                if (length < 1.0e-6f) {
                    continue;
                }
                axis = ra_scale(axis, 1.0f / length);
            }
            float reach = ra_oboxr(
                    cube_axes, cube_half, axis)
                + ra_oboxr(
                    pad_axes, pad.half_extents, axis)
                + margin;
            if (fabsf(ra_dot(delta, axis)) > reach) {
                sat_ok = 0;
                break;
            }
        }
        if (!sat_ok || ra_dot(delta, inward) < 0.0f) {
            continue;
        }
        RaVec3 inner_surface = ra_add(pad.pose.position,
            ra_scale(inward, pad.half_extents.y));
        RaVec3 cube_surface = cube_position;
        RaVec3 support_dir = ra_scale(inward, -1.0f);
        for (int axis = 0; axis < 3; ++axis) {
            float sign = ra_dot(cube_axes[axis], support_dir) < 0.0f
                ? -1.0f : 1.0f;
            cube_surface = ra_add(cube_surface,
                ra_scale(cube_axes[axis], sign * RA_CUBE_HALF));
        }
        float face_separation = ra_dot(
            ra_sub(cube_surface, inner_surface), inward);
        if (face_separation > margin) {
            continue;
        }
        float local_x = ra_dot(delta, pad_axes[0]);
        float local_z = ra_dot(delta, pad_axes[2]);
        float cube_radius_x = ra_oboxr(
            cube_axes, cube_half, pad_axes[0]);
        float cube_radius_z = ra_oboxr(
            cube_axes, cube_half, pad_axes[2]);
        float low_x = ra_max(-pad.half_extents.x,
            local_x - cube_radius_x);
        float high_x = ra_min(pad.half_extents.x,
            local_x + cube_radius_x);
        float low_z = ra_max(-pad.half_extents.z,
            local_z - cube_radius_z);
        float high_z = ra_min(pad.half_extents.z,
            local_z + cube_radius_z);
        float patch_x = low_x <= high_x
            ? 0.5f * (low_x + high_x)
            : ra_clamp(local_x, -pad.half_extents.x, pad.half_extents.x);
        float patch_z = low_z <= high_z
            ? 0.5f * (low_z + high_z)
            : ra_clamp(local_z, -pad.half_extents.z, pad.half_extents.z);
        candidate.hit = 1;
        candidate.iterations = 15;
        candidate.separation = face_separation;
        candidate.normal = inward;
        candidate.point_b = ra_add(inner_surface,
            ra_add(ra_scale(pad_axes[0], patch_x),
                ra_scale(pad_axes[2], patch_z)));
        candidate.point_a = ra_add(candidate.point_b,
            ra_scale(inward, face_separation));
        if (!found || candidate.separation
                < best->separation - 2.0e-5f) {
            *best = candidate;
            point_a_sum = candidate.point_a;
            point_b_sum = candidate.point_b;
            manifold_points = 1;
            found = 1;
        } else if (candidate.separation
                <= best->separation + 2.0e-5f) {
            point_a_sum = ra_add(point_a_sum, candidate.point_a);
            point_b_sum = ra_add(point_b_sum, candidate.point_b);
            manifold_points++;
        }
    }
    if (found) {
        float inverse_points = 1.0f / (float)manifold_points;
        best->point_a = ra_scale(point_a_sum, inverse_points);
        best->point_b = ra_scale(point_b_sum, inverse_points);
        best->normal = inward;
    }
    return found;
}

typedef struct RaGripperCollisionFrame {
    RaPose hand;
    RaPose left_finger;
    RaPose right_finger;
} RaGripperCollisionFrame;

typedef struct RaCollisionBox {
    RaPose pose;
    RaVec3 half_extent;
    int pad_face;
} RaCollisionBox;

RA_D static RA_INLINE RaGripperCollisionFrame ra_gripf(
        const RaPose* links, RaVec3 end_effector) {
    RaGripperCollisionFrame frame;
    frame.hand.rotation = links[RA_DOF + 1].rotation;
    frame.hand.position = ra_sub(end_effector,
        ra_rotate(frame.hand.rotation, ra_v3(0, 0, 0.115f)));
    frame.left_finger = links[RA_DOF + 1];
    frame.right_finger = links[RA_DOF + 2];
    return frame;
}

RA_D static RA_INLINE RaPose ra_offp(
        RaPose parent, RaVec3 local_position) {
    RaPose pose;
    pose.position = ra_add(parent.position,
        ra_rotate(parent.rotation, local_position));
    pose.rotation = parent.rotation;
    return pose;
}

RA_D static RA_INLINE RaCollisionBox ra_linkb(
        const RaPose* links, int index) {
    const RaVec3 center[RA_DOF] = {
        {-0.00001f, -0.03719f, -0.06850f},
        {-0.00001f, -0.06949f,  0.03720f},
        { 0.04124f,  0.02803f, -0.03300f},
        {-0.04126f,  0.03450f,  0.02803f},
        {-0.00001f,  0.03747f, -0.10340f},
        { 0.04206f,  0.01523f,  0.00613f},
        { 0.01864f,  0.01863f,  0.07940f},
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
    RaCollisionBox box;
    int link = index < 0 ? 0 : (index >= RA_DOF ? RA_DOF - 1 : index);
    box.pose = ra_offp(links[link + 1], center[link]);
    box.half_extent = half_extent[link];
    box.pad_face = 0;
    return box;
}

RA_D static RA_INLINE RaCollisionBox ra_gripb(
        const RaGripperCollisionFrame* frame, int index) {
    RaCollisionBox box;
    box.pad_face = 0;
    if (index == 0) {
        box.pose = ra_offp(frame->hand, ra_v3(0, 0, -0.0055f));
        box.half_extent = ra_v3(0.0320f, 0.1040f, 0.0205f);
    } else if (index == 1) {
        box.pose = ra_offp(frame->hand, ra_v3(0, 0, 0.0250f));
        box.half_extent = ra_v3(0.0240f, 0.1020f, 0.0100f);
    } else if (index == 2) {
        box.pose = ra_offp(frame->hand, ra_v3(0, 0, 0.0505f));
        box.half_extent = ra_v3(0.0220f, 0.1010f, 0.0155f);
    } else {
        int right = index >= 5;
        int distal = index == 4 || index == 6;
        RaPose finger = right
            ? frame->right_finger : frame->left_finger;
        if (distal) {
            box.pose = ra_offp(finger, ra_v3(0, 0.0080f, 0.0420f));
            box.half_extent = ra_v3(0.0095f, 0.0080f, 0.0120f);
            box.pad_face = 1;
        } else {
            box.pose = ra_offp(finger, ra_v3(0, 0.0144f, 0.0150f));
            box.half_extent = ra_v3(0.0105f, 0.0120f, 0.0150f);
        }
    }
    return box;
}

RA_HD static RA_INLINE RaVec3 ra_gctr(
        RaVec3 end_effector, RaQuat hand_rotation) {
    return ra_sub(end_effector, ra_rotate(hand_rotation,
        ra_v3(0, 0, RA_BASKETBALL_GRASP_CENTER_OFFSET)));
}

RA_HD static RA_INLINE RaVec3 ra_hoop(void) {
    return ra_v3(RA_HOOP_CENTER_X, RA_HOOP_CENTER_Y, RA_HOOP_CENTER_Z);
}

RA_HD static RA_INLINE RaVec3 ra_bvel(
        RaVec3 velocity, float dt) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    float decay = expf(-drag * dt);
    velocity.x *= decay;
    velocity.y = (velocity.y + gravity / drag) * decay - gravity / drag;
    velocity.z *= decay;
    return velocity;
}

RA_HD static RA_INLINE RaVec3 ra_bpos(
        RaVec3 position, RaVec3 velocity, float time) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    float decay = expf(-drag * time);
    float travel = (1.0f - decay) / drag;
    return ra_v3(
        position.x + velocity.x * travel,
        position.y + (velocity.y + gravity / drag) * travel
            - gravity * time / drag,
        position.z + velocity.z * travel);
}

RA_HD static RA_INLINE float ra_blq(
        RaVec3 position, RaVec3 velocity) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    RaVec3 delta = ra_sub(ra_hoop(), position);
    float horizontal = sqrtf(delta.x*delta.x + delta.z*delta.z);
    float flight_time = ra_clamp(horizontal / 2.20f, 0.45f, 0.75f);
    float travel = (1.0f - expf(-drag * flight_time)) / drag;
    RaVec3 target = ra_v3(delta.x / travel,
        (delta.y + gravity * flight_time / drag) / travel - gravity / drag,
        delta.z / travel);
    RaVec3 error = ra_sub(velocity, target);
    float error_squared = ra_dot(error, error);
    const float sigma = 1.50f;
    return expf(-0.5f * error_squared / (sigma*sigma));
}

RA_HD static RA_INLINE int ra_bxing(
        RaVec3 position, RaVec3 velocity, RaVec3* crossing,
        float* crossing_time, RaVec3* apex) {
    const float gravity = 9.81f;
    const float drag = RA_BALL_LINEAR_DRAG;
    RaVec3 hoop = ra_hoop();
    float apex_time = velocity.y > 0.0f
        ? logf(1.0f + drag * velocity.y / gravity) / drag : 0.0f;
    RaVec3 apex_position = ra_bpos(
        position, velocity, apex_time);
    if (apex != NULL) {
        *apex = apex_position;
    }
    if (position.y < hoop.y && apex_position.y <= hoop.y) {
        return 0;
    }
    float discriminant = velocity.y*velocity.y
        + 2.0f*gravity*(position.y - hoop.y);
    if (discriminant <= 0.0f) {
        return 0;
    }
    float time = (velocity.y + sqrtf(discriminant)) / gravity;
    time = ra_clamp(time, apex_time + 1.0e-4f, 2.0f);
    for (int iteration = 0; iteration < 3; ++iteration) {
        float decay = expf(-drag * time);
        float predicted_y = position.y
            + (velocity.y + gravity / drag) * (1.0f - decay) / drag
            - gravity * time / drag;
        float predicted_vy = (velocity.y + gravity / drag) * decay
            - gravity / drag;
        if (predicted_vy >= -1.0e-4f) {
            return 0;
        }
        time = ra_clamp(time - (predicted_y - hoop.y) / predicted_vy,
            apex_time + 1.0e-4f, 2.0f);
    }
    RaVec3 predicted = ra_bpos(
        position, velocity, time);
    float decay = expf(-drag * time);
    float predicted_vy = (velocity.y + gravity / drag) * decay
        - gravity / drag;
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

RA_HD static RA_INLINE float ra_btq(
        RaVec3 position, RaVec3 velocity) {
    RaVec3 crossing;
    if (!ra_bxing(
            position, velocity, &crossing, NULL, NULL)) {
        return 0.0f;
    }
    RaVec3 hoop = ra_hoop();
    float dx = crossing.x - hoop.x;
    float dz = crossing.z - hoop.z;
    float radial_error_squared = dx*dx + dz*dz;
    const float coarse_sigma = 0.25f;
    const float fine_sigma = 0.055f;
    float coarse = expf(-0.5f * radial_error_squared
        / (coarse_sigma*coarse_sigma));
    float fine = expf(-0.5f * radial_error_squared
        / (fine_sigma*fine_sigma));
    return 0.25f*coarse + 0.75f*fine;
}

RA_D static RA_INLINE void ra_obs_xyz(
        float* observation, int* index, RaVec3 value) {
    observation[(*index)++] = value.x;
    observation[(*index)++] = value.y;
    observation[(*index)++] = value.z;
}

RA_D static RA_INLINE void ra_obs3(
        float* observation, int* index, RaVec3 value, float scale) {
    observation[(*index)++] = ra_clamp(scale * value.x, -1.0f, 1.0f);
    observation[(*index)++] = ra_clamp(scale * value.y, -1.0f, 1.0f);
    observation[(*index)++] = ra_clamp(scale * value.z, -1.0f, 1.0f);
}

RA_D static RA_INLINE float ra_obs_pad(float impulse) {
    return ra_clamp(impulse / (RA_PHYSICS_DT * RA_GRIPPER_MAX_FORCE),
        0.0f, 1.0f);
}

RA_D static void ra_observe(const RaState* state, float* observation) {
    int index = 0;
    for (int joint = 0; joint < RA_DOF; ++joint) {
        float low = ra_jmin(joint);
        float high = ra_jmax(joint);
        float midpoint = 0.5f * (low + high);
        float half_range = 0.5f * (high - low);
        observation[index++] = ra_clamp(
            (state->q[joint] - midpoint) / half_range, -1.0f, 1.0f);
    }
    for (int joint = 0; joint < RA_DOF; ++joint) {
        observation[index++] = ra_clamp(state->qd[joint] / 6.0f, -1.0f, 1.0f);
    }
    for (int action = 0; action < RA_ACTIONS; ++action) {
        observation[index++] = ra_clamp(
            state->previous_action[action], -1.0f, 1.0f);
    }
    RaPose gripper_links[RA_LINKS];
    RaVec3 origins[RA_DOF];
    RaVec3 axes[RA_DOF];
    ra_fk(state->q, state->gripper_width, gripper_links, origins, axes, NULL);
    RaQuat hand_rotation = gripper_links[RA_DOF + 1].rotation;
    RaVec3 reach_origin = state->basketball_mode
        ? ra_gctr(state->end_effector, hand_rotation)
        : state->end_effector;
    ra_obs3(observation, &index,
        ra_sub(state->cube_position, reach_origin), RA_OBS_POS_SCALE);
    ra_obs3(observation, &index,
        ra_sub(state->target_position, state->cube_position),
        RA_OBS_POS_SCALE);
    ra_obs3(observation, &index, state->cube_velocity, RA_OBS_LIN_VEL_SCALE);
    ra_obs3(observation, &index, state->end_effector, RA_OBS_POS_SCALE);
    RaQuat gripper_in_cube = state->basketball_mode
        ? hand_rotation
        : ra_qmul(ra_qconj(state->cube_rotation), hand_rotation);
    ra_obs_xyz(observation, &index,
        ra_rotate(gripper_in_cube, ra_v3(0, 1, 0)));
    ra_obs_xyz(observation, &index,
        ra_rotate(gripper_in_cube, ra_v3(0, 0, 1)));
    ra_obs3(observation, &index, state->cube_angular_velocity,
        RA_OBS_ANG_VEL_SCALE);
    float grip_vel = ra_clamp(
        RA_OBS_GRIP_VEL_SCALE * state->gripper_velocity, -1.0f, 1.0f);
    if (state->stack_mode) {
        ra_obs_xyz(observation, &index,
            ra_rotate(state->base_cube_rotation, ra_v3(1, 0, 0)));
        ra_obs_xyz(observation, &index,
            ra_rotate(state->base_cube_rotation, ra_v3(0, 1, 0)));
        ra_obs3(observation, &index, state->base_cube_velocity,
            RA_OBS_LIN_VEL_SCALE);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        observation[index++] = ra_clamp(
            RA_OBS_ANG_VEL_SCALE * ra_length(state->base_cube_angular_velocity),
            0.0f, 1.0f);
    } else if (state->basketball_mode) {
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        observation[index++] = grip_vel;
        ra_obs3(observation, &index, state->cube_position, RA_OBS_POS_SCALE);
        ra_obs3(observation, &index, ra_rotate(
            ra_qconj(hand_rotation), state->cube_velocity),
            RA_OBS_LIN_VEL_SCALE);
        ra_obs3(observation, &index, ra_ptvel(
            state->qd, origins, axes, RA_DOF - 1, state->end_effector),
            RA_OBS_LIN_VEL_SCALE);
    } else {
        ra_obs3(observation, &index, state->target_position, RA_OBS_POS_SCALE);
        ra_obs_xyz(observation, &index,
            ra_rotate(state->cube_rotation, ra_v3(0, 1, 0)));
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[0]);
        observation[index++] = ra_obs_pad(state->pad_normal_impulse[1]);
        ra_obs_xyz(observation, &index,
            ra_rotate(hand_rotation, ra_v3(0, 0, 1)));
        observation[index++] = grip_vel;
    }
    observation[index++] = ra_clamp(
        state->gripper_width / 0.08f, 0.0f, 1.0f);
    observation[index++] = ra_clamp(
        state->gripper_force / RA_GRIPPER_MAX_FORCE, 0.0f, 1.0f);
    ra_obs3(observation, &index, ra_rotate(
        ra_qconj(hand_rotation),
        ra_scale(state->wrist_linear_impulse, 1.0f / RA_CONTROL_DT)), 0.01f);
    ra_obs3(observation, &index, ra_rotate(
        ra_qconj(hand_rotation),
        ra_scale(state->wrist_angular_impulse, 1.0f / RA_CONTROL_DT)), 0.20f);
    observation[index++] = state->transported ? 1.0f : 0.0f;
    observation[index++] = state->basketball_mode
        ? (state->basketball_close_ready ? 1.0f : 0.0f)
        : (state->stack_aligned ? 1.0f : 0.0f);
    observation[index++] = state->basketball_mode
        ? (state->basketball_in_flight ? 1.0f : 0.0f)
        : (state->released_near_target ? 1.0f : 0.0f);
    observation[index++] = state->grasped ? 1.0f : 0.0f;
    observation[index++] = state->lifted ? 1.0f : 0.0f;
    int maximum_steps = state->basketball_mode
        ? RA_BASKETBALL_MAX_STEPS : RA_MAX_STEPS;
    observation[index++] = ra_clamp(
        (float)state->step / (float)maximum_steps, 0.0f, 1.0f);
    assert(index == OBS_SIZE);
}

RA_HD static void ra_resetb(RaState* state) {
    state->cube_position = ra_v3(
        ra_rand(&state->rng, 0.42f, 0.54f),
        RA_TABLE_TOP + RA_BALL_RADIUS,
        ra_rand(&state->rng, 0.20f, 0.32f));
    state->cube_velocity = ra_v3(0, 0, 0);
    state->cube_rotation = ra_quat(0, 0, 0, 1);
    state->cube_angular_velocity = ra_v3(0, 0, 0);
    state->previous_ball_position = state->cube_position;
    state->target_position = ra_hoop();
    state->basketball_in_flight = 0;
    state->basketball_grounded_steps = 0;
    state->grasped = 0;
    state->grasp_cooldown = 0;
    state->grasp_contact_misses = 0;
    state->ever_grasped = 0;
    state->lifted = 0;
    state->transported = 0;
    state->released_near_target = 0;
    state->placement_settle_steps = 0;
    state->basketball_close_ready = 0;
    state->basketball_release_ready = 0;
    state->basketball_release_commanded = 0;
    state->gripper_force = 0.0f;
    memset(state->pad_normal_impulse, 0,
        sizeof(state->pad_normal_impulse));
    RaPose links[RA_LINKS];
    ra_fk(state->q, state->gripper_width, links, NULL, NULL, NULL);
    RaVec3 grasp_center = ra_gctr(
        state->end_effector, links[RA_DOF + 1].rotation);
    state->previous_reach_distance = ra_length(
        ra_sub(state->cube_position, grasp_center));
    state->previous_place_distance = ra_length(
        ra_sub(state->target_position, state->cube_position));
    state->previous_lift_height = 0.0f;
    state->previous_grip_error = fabsf(
        state->gripper_width - RA_BASKETBALL_OPEN_WIDTH);
    float launch_quality = ra_blq(
        state->cube_position, state->cube_velocity);
    float trajectory_quality = ra_btq(
        state->cube_position, state->cube_velocity);
    state->previous_throw_quality = 0.35f*launch_quality
        + 0.65f*trajectory_quality;
}

RA_HD static void ra_reset(RaState* state) {
    uint32_t rng = state->rng ? state->rng : 1u;
    int no_timeout = state->no_timeout;
    int stack_mode = state->stack_mode;
    int basketball_mode = state->basketball_mode;
    memset(state, 0, sizeof(*state));
    state->rng = rng;
    state->no_timeout = no_timeout;
    state->stack_mode = stack_mode;
    state->basketball_mode = basketball_mode;
    state->cube_rotation = ra_quat(0, 0, 0, 1);
    state->base_cube_rotation = ra_quat(0, 0, 0, 1);
    for (int joint = 0; joint < RA_DOF; ++joint) {
        state->q[joint] = ra_jhome(joint)
            + ra_rand(&state->rng, -0.035f, 0.035f);
        state->target_q[joint] = state->q[joint];
    }
    state->gripper_width = 0.080f;
    RaPose links[RA_LINKS];
    ra_fk(state->q, state->gripper_width, links, NULL, NULL,
        &state->end_effector);

    if (state->basketball_mode) {
        ra_resetb(state);
        return;
    }

    float cube_angle = ra_rand(&state->rng, -0.72f, -0.28f);
    float cube_radius = ra_rand(&state->rng, 0.43f, 0.62f);
    state->cube_position = ra_v3(cube_radius*cosf(cube_angle),
        RA_TABLE_TOP + RA_CUBE_HALF, -cube_radius*sinf(cube_angle));
    float target_angle = ra_rand(&state->rng, 0.28f, 0.72f);
    float target_radius = ra_rand(&state->rng, 0.43f, 0.62f);
    if (state->stack_mode) {
        state->base_cube_position = ra_v3(target_radius*cosf(target_angle),
            RA_TABLE_TOP + RA_CUBE_HALF, -target_radius*sinf(target_angle));
        state->base_cube_start_position = state->base_cube_position;
        state->previous_base_cube_position = state->base_cube_position;
        state->target_position = ra_add(state->base_cube_position,
            ra_v3(0, 2.0f * RA_CUBE_HALF, 0));
    } else {
        state->target_position = ra_v3(target_radius*cosf(target_angle),
            RA_TABLE_TOP + 0.008f, -target_radius*sinf(target_angle));
    }
    state->previous_reach_distance = ra_length(
        ra_sub(state->cube_position, state->end_effector));
    state->previous_place_distance = ra_length(
        ra_sub(state->target_position, state->cube_position));
    state->previous_lift_height = 0.0f;
    RaVec3 stack_delta = ra_sub(
        state->cube_position, state->base_cube_position);
    state->previous_stack_horizontal = sqrtf(
        stack_delta.x*stack_delta.x + stack_delta.z*stack_delta.z);
    float stack_clearance = stack_delta.y - 2.0f*RA_CUBE_HALF;
    state->previous_stack_drop_error = fabsf(
        stack_clearance - RA_STACK_HOVER_CLEARANCE);
    state->previous_stack_orientation_error = 0.0f;
}

RA_D static void ra_logep(const RaState* state, struct Log* log) {
    if (state->basketball_mode) {
        float grasp_denominator = ra_max(
            (float)state->attempts, (float)state->basketball_grasps);
        float release_denominator = ra_max(
            (float)state->basketball_grasps, 1.0f);
        log->basketball_mode += 1.0f;
        log->score += (float)state->baskets;
        log->baskets += (float)state->baskets;
        log->grasp_rate += grasp_denominator > 0.0f
            ? (float)state->basketball_grasps / grasp_denominator : 0.0f;
        log->lift_rate += state->lifted ? 1.0f : 0.0f;
        log->slip_rate += state->slip_events > 0 ? 1.0f : 0.0f;
        log->release_rate += (float)state->basketball_releases
            / release_denominator;
        log->release_center_miss_cm_sum +=
            state->basketball_release_center_miss_cm_sum;
        log->release_center_miss_count +=
            (float)state->basketball_releases;
        log->episode_length += (float)state->step;
        log->n += 1.0f;
        return;
    }
    log->score += state->success ? 1.0f : 0.0f;
    log->episode_length += (float)state->step;
    log->success_rate += state->success ? 1.0f : 0.0f;
    log->grasp_rate += state->ever_grasped ? 1.0f : 0.0f;
    log->lift_rate += state->lifted ? 1.0f : 0.0f;
    log->transport_rate += state->transported ? 1.0f : 0.0f;
    log->release_rate += state->stack_mode
        ? (state->valid_release_achieved ? 1.0f : 0.0f)
        : (state->released_near_target ? 1.0f : 0.0f);
    log->return_value += state->episode_return;
    log->reach_distance += ra_length(
        ra_sub(state->cube_position, state->end_effector));
    log->place_distance += ra_length(
        ra_sub(state->target_position, state->cube_position));
    log->energy += state->episode_energy / ra_max((float)state->step, 1.0f);
    log->pinch_force += state->episode_pinch_force
        / ra_max((float)state->pinch_substeps, 1.0f);
    log->slip_rate += state->slip_events > 0 ? 1.0f : 0.0f;
    log->cube_angular_speed += ra_length(state->cube_angular_velocity);
    log->base_angular_speed += state->stack_mode
        ? ra_length(state->base_cube_angular_velocity) : 0.0f;
    log->orientation_error += state->stack_mode
        ? ra_max(ra_cup(state->cube_rotation),
            ra_cup(state->base_cube_rotation))
        : ra_cup(state->cube_rotation);
    if (state->stack_mode) {
        RaVec3 alignment = ra_sub(
            state->cube_position, state->base_cube_position);
        RaVec3 base_slide = ra_sub(
            state->base_cube_position, state->base_cube_start_position);
        log->stack_rate += state->ever_stacked ? 1.0f : 0.0f;
        log->stable_stack_rate += state->success ? 1.0f : 0.0f;
        log->stack_alignment_rate += state->stack_aligned ? 1.0f : 0.0f;
        log->valid_stack_contact_rate += state->valid_stack_contact
            ? 1.0f : 0.0f;
        log->clearance_rate += state->cleared_after_release ? 1.0f : 0.0f;
        log->settle_rate += state->max_placement_settle_steps > 0
            ? 1.0f : 0.0f;
        log->stack_alignment += sqrtf(
            alignment.x*alignment.x + alignment.z*alignment.z);
        log->base_slide_distance += sqrtf(
            base_slide.x*base_slide.x + base_slide.z*base_slide.z);
    }
    log->n += 1.0f;
}

RA_D static float ra_stepb(RaState* state,
        const float* actions, float energy, int first_grasp, int released,
        const RaPose* links) {
    RaVec3 hoop = ra_hoop();
    RaVec3 grasp_center = ra_gctr(
        state->end_effector, links[RA_DOF + 1].rotation);
    float reach_distance = ra_length(
        ra_sub(state->cube_position, grasp_center));
    float hoop_distance = ra_length(ra_sub(hoop, state->cube_position));
    float lift_height = ra_max(0.0f,
        state->cube_position.y - RA_BALL_RADIUS - RA_TABLE_TOP);
    float reward = -0.0001f;

    if (!state->grasped && !state->basketball_in_flight) {
        reward += 0.08f * ra_clamp(
            state->previous_reach_distance - reach_distance,
            -0.05f, 0.05f);
    } else if (state->grasped && !state->lifted) {
        reward += 0.08f * ra_clamp(
            lift_height - state->previous_lift_height, -0.03f, 0.03f);
    } else if (state->grasped) {
        reward += 0.04f * ra_clamp(
            state->previous_place_distance - hoop_distance,
            -0.05f, 0.05f);
    } else if (state->basketball_in_flight) {
        reward += 0.02f * ra_clamp(
            state->previous_place_distance - hoop_distance,
            -0.05f, 0.05f);
    }
    int open_enough = state->gripper_width > 0.062f;
    int entered_close_phase = !state->basketball_close_ready
        && !state->grasped && !state->basketball_in_flight
        && open_enough && reach_distance < 0.045f;
    if (entered_close_phase) {
        state->basketball_close_ready = 1;
        state->previous_grip_error = fabsf(
            state->gripper_width - RA_BASKETBALL_GRIP_WIDTH);
    }
    float target_width = state->basketball_close_ready
        ? RA_BASKETBALL_GRIP_WIDTH : RA_BASKETBALL_OPEN_WIDTH;
    float grip_error = fabsf(state->gripper_width - target_width);
    if (!state->grasped && !state->basketball_in_flight
            && !entered_close_phase) {
        reward += 0.15f * ra_clamp(
            state->previous_grip_error - grip_error, -0.02f, 0.02f);
    }
    if (first_grasp) {
        state->basketball_grasps += 1;
        reward += 0.050f;
    }
    if (!state->lifted && state->grasped && lift_height >= RA_LIFT_HEIGHT) {
        state->lifted = 1;
        reward += 0.025f;
    }
    if (!state->transported && state->grasped && state->lifted
            && hoop_distance < RA_BASKETBALL_RELEASE_DISTANCE) {
        state->transported = 1;
        reward += 0.020f;
    }
    float launch_quality = ra_blq(
        state->cube_position, state->cube_velocity);
    float trajectory_quality = ra_btq(
        state->cube_position, state->cube_velocity);
    float release_quality = 0.35f*launch_quality
        + 0.65f*trajectory_quality;
    if (state->grasped && state->lifted) {
        reward += 0.040f * ra_clamp(
            release_quality - state->previous_throw_quality, -0.05f, 0.05f);
        if (!state->basketball_release_ready
                && release_quality >= RA_BASKETBALL_RELEASE_READY_QUALITY) {
            state->basketball_release_ready = 1;
        }
        if (state->basketball_release_ready) {
            reward -= 0.0005f;
            if (actions[RA_DOF] > 0.25f) {
                reward += 0.010f*release_quality;
            }
        }
    }
    int opening_for_release = state->lifted && state->ever_grasped
        && !state->basketball_in_flight && actions[RA_DOF] > 0.25f
        && (state->grasped || released);
    if (opening_for_release && !state->basketball_release_commanded) {
        state->basketball_release_commanded = 1;
        reward += 0.015f + 0.015f*release_quality;
    }
    int thrown = state->lifted && !state->grasped
        && state->ever_grasped && released;
    if (!state->basketball_in_flight && thrown) {
        state->basketball_in_flight = 1;
        state->basketball_releases += 1;
        RaVec3 predicted_crossing;
        float center_miss = RA_BASKETBALL_PREDICTED_MISS_CAP;
        if (ra_bxing(
                state->cube_position, state->cube_velocity,
                &predicted_crossing, NULL, NULL)) {
            float dx = predicted_crossing.x - hoop.x;
            float dz = predicted_crossing.z - hoop.z;
            center_miss = ra_min(sqrtf(dx*dx + dz*dz),
                RA_BASKETBALL_PREDICTED_MISS_CAP);
        }
        state->basketball_release_center_miss_cm_sum += 100.0f*center_miss;
        state->basketball_release_ready = 0;
        reward += 0.020f*launch_quality
            + 0.100f*trajectory_quality;
    }

    int crossed_down = state->basketball_in_flight
        && state->previous_ball_position.y > hoop.y
        && state->cube_position.y <= hoop.y;
    int scored = 0;
    if (crossed_down) {
        float height_delta = state->previous_ball_position.y
            - state->cube_position.y;
        float fraction = height_delta > 1.0e-7f
            ? (state->previous_ball_position.y - hoop.y) / height_delta
            : 0.0f;
        RaVec3 crossing = ra_add(state->previous_ball_position,
            ra_scale(ra_sub(state->cube_position,
                state->previous_ball_position), fraction));
        float offset_x = crossing.x - hoop.x;
        float offset_z = crossing.z - hoop.z;
        float clearance = RA_HOOP_INNER_RADIUS - RA_BALL_RADIUS;
        scored = offset_x*offset_x + offset_z*offset_z
            < clearance*clearance;
    }
    int grounded = !state->grasped
        && state->cube_position.y <= RA_TABLE_TOP + RA_BALL_RADIUS
            + RA_BASKETBALL_GROUNDED_HEIGHT_SLOP
        && fabsf(state->cube_velocity.y)
            <= RA_BASKETBALL_GROUNDED_MAX_VERTICAL_SPEED;
    float ball_base_distance = ra_length(state->cube_position);
    int out_of_reach = ball_base_distance - RA_BALL_RADIUS
        > RA_ARM_GEOMETRIC_REACH_BOUND;
    if (grounded && out_of_reach) {
        state->basketball_grounded_steps += 1;
    } else {
        state->basketball_grounded_steps = 0;
    }
    int grounded_reset = state->basketball_grounded_steps
        >= RA_BASKETBALL_GROUNDED_RESET_STEPS;
    if (scored) {
        state->baskets += 1;
        state->attempts += 1;
        state->success = 1;
        reward = 1.0f;
        state->basketball_in_flight = 0;
    } else if (grounded && state->basketball_in_flight) {
        state->attempts += 1;
        reward = -0.010f;
        state->basketball_in_flight = 0;
        state->basketball_release_ready = 0;
    }
    if (grounded_reset) {
        ra_resetb(state);
        state->basketball_reset = 1;
    }

    if (!state->no_timeout && state->step >= RA_BASKETBALL_MAX_STEPS) {
        state->done = 1;
    }
    if (!state->basketball_reset) {
        state->previous_reach_distance = reach_distance;
        state->previous_place_distance = hoop_distance;
        state->previous_lift_height = lift_height;
        state->previous_grip_error = grip_error;
        state->previous_throw_quality = release_quality;
    }
    state->episode_energy += energy;
    state->episode_return += reward;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        state->previous_action[action] = ra_clamp(
            actions[action], -1.0f, 1.0f);
    }
    return reward;
}

RA_D static float ra_stept(RaState* state, const float* actions,
        float energy, int first_grasp, int released, const RaPose* links) {
    float grip_action = ra_clamp(actions[RA_DOF], -1.0f, 1.0f);
    float reach_distance = ra_length(
        ra_sub(state->cube_position, state->end_effector));
    RaQuat hand_rotation = links[RA_DOF + 1].rotation;
    RaVec3 hand_position = ra_sub(state->end_effector,
        ra_rotate(hand_rotation, ra_v3(0, 0, 0.115f)));
    float half_width = 0.5f * state->gripper_width;
    RaPose fingers[2];
    fingers[0].position = ra_add(hand_position,
        ra_rotate(hand_rotation, ra_v3(0, half_width, 0.0584f)));
    fingers[0].rotation = hand_rotation;
    fingers[1].position = ra_add(hand_position,
        ra_rotate(hand_rotation, ra_v3(0, -half_width, 0.0584f)));
    fingers[1].rotation = ra_qnorm(ra_qmul(hand_rotation,
        ra_qaxis(ra_v3(0, 0, 1), 3.14159265359f)));
    RaConvexContact clear_contact;
    int gripper_clear = 1;
    for (int finger = 0; finger < 2; ++finger) {
        gripper_clear &= !ra_padhit(
            state->cube_position, state->cube_rotation,
            fingers[finger], RA_GRIPPER_CLEARANCE_MARGIN, &clear_contact);
        if (state->stack_mode) {
            gripper_clear &= !ra_padhit(
                state->base_cube_position, state->base_cube_rotation,
                fingers[finger], RA_GRIPPER_CLEARANCE_MARGIN, &clear_contact);
        }
    }
    RaVec3 place_offset = ra_sub(
        state->target_position, state->cube_position);
    float place_distance = ra_length(place_offset);
    float place_horizontal_distance = sqrtf(
        place_offset.x*place_offset.x + place_offset.z*place_offset.z);
    float main_support_y = ra_csup(
        state->cube_rotation, ra_v3(0, 1, 0));
    float base_support_y = ra_csup(
        state->base_cube_rotation, ra_v3(0, 1, 0));
    float expected_stack_separation = main_support_y + base_support_y;
    float stack_height_error = fabsf(
        (state->cube_position.y - state->base_cube_position.y)
            - expected_stack_separation);
    float stack_clearance = (state->cube_position.y
        - state->base_cube_position.y) - expected_stack_separation;
    float stack_drop_error = fabsf(
        stack_clearance - RA_STACK_HOVER_CLEARANCE);
    float stack_orientation_error = ra_cup(
        state->cube_rotation);
    RaVec3 base_motion_delta = ra_sub(
        state->base_cube_position, state->previous_base_cube_position);
    float base_motion = sqrtf(base_motion_delta.x*base_motion_delta.x
        + base_motion_delta.z*base_motion_delta.z);
    int stack_release_pose = place_horizontal_distance
            < RA_STACK_RELEASE_RADIUS
        && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE
        && stack_clearance < RA_STACK_RELEASE_CLEARANCE;
    int stack_alignment_pose = place_horizontal_distance < 0.050f
        && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE
        && stack_clearance < 0.080f;
    float lift_height = ra_max(0.0f,
        state->cube_position.y - main_support_y - RA_TABLE_TOP);
    int stack_contact = state->stack_mode && !state->grasped
        && place_horizontal_distance < 2.0f*RA_CUBE_HALF
        && stack_height_error < RA_STACK_HEIGHT_TOLERANCE;
    if (stack_contact) {
        state->ever_stacked = 1;
    }
    int placement_disturbed = state->released_near_target
        && (state->grasped
            || place_horizontal_distance >= (state->stack_mode
                ? RA_STACK_RELEASE_RADIUS : RA_PLACE_RADIUS));
    if (placement_disturbed) {
        state->released_near_target = 0;
        state->placement_settle_steps = 0;
    }
    float reward = -0.002f;
    if (!state->grasped) {
        if (state->released_near_target) {
            reward += (state->stack_mode ? 8.0f : 1.8f) * ra_clamp(
                reach_distance - state->previous_reach_distance,
                -0.05f, 0.05f);
        } else {
            reward += 1.8f * ra_clamp(
                state->previous_reach_distance - reach_distance,
                -0.05f, 0.05f);
        }
    } else if (!state->lifted) {
        reward += (state->stack_mode ? 12.0f : 5.0f) * ra_clamp(
            lift_height - state->previous_lift_height, -0.03f, 0.03f);
        if (lift_height >= RA_LIFT_HEIGHT) {
            state->lifted = 1;
            reward += state->stack_mode ? 2.0f : 0.75f;
        }
    }
    if (state->grasped && state->lifted) {
        if (state->stack_mode) {
            reward += RA_STACK_HORIZONTAL_PROGRESS_REWARD * ra_clamp(
                state->previous_stack_horizontal
                    - place_horizontal_distance,
                -0.05f, 0.05f);
            if (place_horizontal_distance < 0.12f) {
                reward += RA_STACK_HEIGHT_PROGRESS_REWARD * ra_clamp(
                    state->previous_stack_drop_error - stack_drop_error,
                    -0.04f, 0.04f);
                reward += RA_STACK_UPRIGHT_PROGRESS_REWARD * ra_clamp(
                    state->previous_stack_orientation_error
                        - stack_orientation_error,
                    -0.05f, 0.05f);
            }
        } else {
            reward += 6.0f * ra_clamp(
                state->previous_place_distance - place_distance,
                -0.05f, 0.05f);
        }
    }
    if (first_grasp) {
        reward += state->stack_mode ? 0.40f : 0.5f;
    }
    if (released && state->stack_mode) {
        if (grip_action <= 0.25f) {
            float slip_penalty = state->stack_aligned ? 2.00f
                : (state->transported ? 1.00f
                : (state->lifted ? RA_STACK_SLIP_PENALTY : 0.10f));
            reward -= slip_penalty;
        } else if (!state->transported) {
            reward -= state->lifted ? 0.75f : 0.15f;
        }
    } else if (released && !state->transported) {
        reward -= state->lifted ? 0.50f : 0.10f;
    }
    if (placement_disturbed) {
        reward -= 0.25f;
    }
    int transport_pose = state->stack_mode
        ? (place_horizontal_distance < RA_STACK_TRANSPORT_RADIUS
            && stack_clearance >= -RA_STACK_HEIGHT_TOLERANCE
            && stack_clearance < 0.10f)
        : place_distance < 0.12f;
    if (!state->transported && state->grasped && state->lifted
            && lift_height >= RA_CARRY_HEIGHT && transport_pose) {
        state->transported = 1;
        reward += state->stack_mode ? 2.0f : 0.5f;
    }
    int stack_release_ready = stack_release_pose
        && stack_orientation_error < 0.050f
        && ra_length(state->cube_velocity) < 0.50f
        && ra_length(state->cube_angular_velocity) < 2.0f;
    if (state->stack_mode && state->transported && state->grasped
            && stack_release_ready && !state->stack_aligned) {
        state->stack_aligned = 1;
        reward += 2.0f;
    }
    if (state->stack_mode && state->transported && state->stack_aligned
            && state->grasped) {
        reward -= 0.030f;
        if (!state->stack_opening_credited && grip_action > 0.25f) {
            state->stack_opening_credited = 1;
            if (stack_release_ready) {
                reward += 1.000f;
            } else if (stack_alignment_pose) {
                reward += 0.200f;
            }
        }
    }
    int valid_release = released && (state->stack_mode
        ? (grip_action > 0.25f && state->stack_aligned
            && stack_alignment_pose)
        : place_distance < 0.12f);
    int first_valid_release = !state->valid_release_achieved
        && state->transported && valid_release;
    if (!state->released_near_target && state->transported && valid_release) {
        state->released_near_target = 1;
        state->valid_release_achieved = 1;
        if (state->stack_mode && first_valid_release) {
            float release_quality_penalty = ra_clamp(
                2.0f*ra_length(state->cube_velocity)
                    + 0.5f*ra_length(state->cube_angular_velocity)
                    + 20.0f*stack_orientation_error,
                0.0f, 4.0f);
            reward += 6.0f - release_quality_penalty;
        } else if (!state->stack_mode) {
            reward += 0.25f;
        }
    }
    if (state->stack_mode && state->transported && released
            && !valid_release && grip_action > 0.25f) {
        reward -= 2.0f;
    }
    if (state->stack_mode && stack_contact
            && state->valid_release_achieved
            && !state->valid_stack_contact) {
        state->valid_stack_contact = 1;
        reward += 4.0f;
    }
    if (state->stack_mode && state->released_near_target
            && gripper_clear
            && !state->cleared_after_release) {
        state->cleared_after_release = 1;
        reward += 4.0f;
    }
    if (state->stack_mode && state->released_near_target
            && place_horizontal_distance < 0.040f
            && stack_height_error < 0.010f) {
        if (stack_orientation_error < RA_STACK_UPRIGHT_ERROR) {
            reward += 0.020f;
        }
        if (ra_length(state->cube_velocity) < RA_STACK_SETTLE_SPEED
                && ra_length(state->base_cube_velocity)
                    < RA_STACK_SETTLE_SPEED) {
            reward += 0.020f;
        }
        if (ra_length(state->cube_angular_velocity)
                < RA_STACK_SETTLE_ANGULAR_SPEED
                && ra_length(state->base_cube_angular_velocity)
                    < RA_STACK_SETTLE_ANGULAR_SPEED) {
            reward += 0.020f;
        }
        if (gripper_clear) {
            reward += 0.030f;
        }
    }
    if (state->stack_mode) {
        reward -= 8.0f * ra_min(base_motion, 0.03f);
    }
    float action_cost = 0.0f;
    float action_delta = 0.0f;
    for (int action = 0; action < RA_DOF; ++action) {
        float value = ra_clamp(actions[action], -1.0f, 1.0f);
        action_cost += value * value;
        float delta = value - state->previous_action[action];
        action_delta += delta * delta;
    }

    int placement_stable;
    int settle_steps_required;
    if (state->stack_mode) {
        placement_stable = !state->grasped
            && state->released_near_target
            && state->lifted && state->transported
            && place_horizontal_distance < RA_STACK_ALIGN_RADIUS
            && stack_height_error < RA_STACK_HEIGHT_TOLERANCE
            && ra_cup(state->cube_rotation)
                < RA_STACK_UPRIGHT_ERROR
            && ra_cup(state->base_cube_rotation)
                < RA_STACK_UPRIGHT_ERROR
            && state->base_cube_position.y
                <= RA_TABLE_TOP + base_support_y + 0.004f
            && ra_length(state->cube_velocity) < RA_STACK_SETTLE_SPEED
            && ra_length(state->base_cube_velocity) < RA_STACK_SETTLE_SPEED
            && ra_length(state->cube_angular_velocity)
                < RA_STACK_SETTLE_ANGULAR_SPEED
            && ra_length(state->base_cube_angular_velocity)
                < RA_STACK_SETTLE_ANGULAR_SPEED
            && gripper_clear;
        settle_steps_required = RA_STACK_SETTLE_STEPS;
    } else {
        placement_stable = !state->grasped
            && state->released_near_target
            && state->lifted && state->transported
            && place_horizontal_distance < RA_PLACE_RADIUS
            && state->cube_position.y
                <= RA_TABLE_TOP + main_support_y + 0.008f
            && ra_length(state->cube_velocity) < RA_PLACE_SETTLE_SPEED
            && ra_length(state->cube_angular_velocity)
                < RA_PLACE_SETTLE_ANGULAR_SPEED
            && reach_distance > RA_PLACE_CLEARANCE;
        settle_steps_required = RA_PLACE_SETTLE_STEPS;
    }
    if (placement_stable) {
        state->placement_settle_steps += 1;
        if (state->placement_settle_steps
                > state->max_placement_settle_steps) {
            state->max_placement_settle_steps
                = state->placement_settle_steps;
        }
        reward += state->stack_mode ? 0.12f : 0.01f;
    } else {
        state->placement_settle_steps = 0;
    }
    if (state->placement_settle_steps >= settle_steps_required) {
        state->success = 1;
        state->done = 1;
        reward += state->stack_mode ? 20.0f : 10.0f;
    }
    if (!state->no_timeout && state->step >= RA_MAX_STEPS) {
        state->done = 1;
    }
    if (state->cube_position.y < -0.25f) {
        state->done = 1;
        reward -= 0.25f;
    }
    if (state->stack_mode && state->base_cube_position.y < -0.25f) {
        state->done = 1;
        reward -= 0.25f;
    }

    reward *= state->stack_mode ? RA_STACK_REWARD_SCALE : RA_PICK_REWARD_SCALE;
    reward -= 0.001f * action_cost + 0.0004f * action_delta;
    reward -= 0.00002f * energy;

    state->previous_reach_distance = reach_distance;
    state->previous_place_distance = place_distance;
    state->previous_lift_height = lift_height;
    state->previous_stack_horizontal = place_horizontal_distance;
    state->previous_stack_drop_error = stack_drop_error;
    state->previous_stack_orientation_error = stack_orientation_error;
    state->previous_base_cube_position = state->base_cube_position;
    state->episode_energy += energy;
    state->episode_return += reward;
    for (int action = 0; action < RA_ACTIONS; ++action) {
        state->previous_action[action] = ra_clamp(actions[action], -1.0f, 1.0f);
    }
    return reward;
}

#define RA_EXPECTED_MESHES 11

typedef struct RaRenderer {
    Model arm;
    Model cube;
    Shader skin_shader;
    Matrix inverse_bind[RA_LINKS];
    int light_direction_loc;
    int view_position_loc;
    int skin_shader_loaded;
    int attempted;
    int loaded;
    int cube_loaded;
} RaRenderer;

typedef struct RaRenderHost {
    RaRenderer* renderer;
    const char* model_glb;
    Camera3D camera;
    float camera_yaw;
    float camera_pitch;
    float camera_distance;
    int camera_initialized;
    int reset_requested;
} RaRenderHost;

static Matrix ra_matrix(RaPose pose) {
    Matrix rotation = QuaternionToMatrix((Quaternion){
        pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w});
    Matrix translation = MatrixTranslate(
        pose.position.x, pose.position.y, pose.position.z);
    return MatrixMultiply(rotation, translation);
}

static Vector3 ra_vector3(RaVec3 value) {
    return (Vector3){value.x, value.y, value.z};
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->score);
    if (log->basketball_mode > 0.5f) {
        dict_set(out, "baskets", log->baskets);
        dict_set(out, "grasp_rate", log->grasp_rate);
        dict_set(out, "lift_rate", log->lift_rate);
        dict_set(out, "release_rate", log->release_rate);
        dict_set(out, "slip_rate", log->slip_rate);
        float release_count = log->release_center_miss_count;
        dict_set(out, "avg_release_miss_cm", release_count > 0.0f
            ? log->release_center_miss_cm_sum / release_count : 0.0f);
        dict_set(out, "episode_length", log->episode_length);
        return;
    }
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "grasp_rate", log->grasp_rate);
    dict_set(out, "lift_rate", log->lift_rate);
    dict_set(out, "transport_rate", log->transport_rate);
    dict_set(out, "release_rate", log->release_rate);
    dict_set(out, "episode_return", log->return_value);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reach_distance", log->reach_distance);
    dict_set(out, "place_distance", log->place_distance);
    dict_set(out, "energy", log->energy);
    dict_set(out, "pinch_force", log->pinch_force);
    dict_set(out, "slip_rate", log->slip_rate);
    dict_set(out, "stack_rate", log->stack_rate);
    dict_set(out, "stable_stack_rate", log->stable_stack_rate);
    dict_set(out, "stack_alignment_rate", log->stack_alignment_rate);
    dict_set(out, "valid_stack_contact_rate",
        log->valid_stack_contact_rate);
    dict_set(out, "clearance_rate", log->clearance_rate);
    dict_set(out, "settle_rate", log->settle_rate);
    dict_set(out, "stack_alignment", log->stack_alignment);
    dict_set(out, "base_slide_distance", log->base_slide_distance);
    dict_set(out, "cube_angular_speed", log->cube_angular_speed);
    dict_set(out, "base_angular_speed", log->base_angular_speed);
    dict_set(out, "orientation_error", log->orientation_error);
}

static void ra_drawc(const RaRenderer* renderer,
        RaVec3 position, RaQuat rotation, Color color, Color wire_color) {
    Quaternion quaternion = (Quaternion){
        rotation.x, rotation.y, rotation.z, rotation.w};
    Vector3 axis;
    float angle;
    QuaternionToAxisAngle(quaternion, &axis, &angle);
    if (axis.x*axis.x + axis.y*axis.y + axis.z*axis.z < 1.0e-8f) {
        axis = (Vector3){0, 1, 0};
    }
    if (renderer->cube_loaded) {
        DrawModelEx(renderer->cube, ra_vector3(position), axis,
            angle * RAD2DEG, (Vector3){1, 1, 1}, color);
    } else {
        DrawCubeV(ra_vector3(position),
            (Vector3){2*RA_CUBE_HALF, 2*RA_CUBE_HALF, 2*RA_CUBE_HALF},
            color);
    }
    static const unsigned char edges[12][2] = {
        {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5}, {2, 3},
        {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7},
    };
    for (int edge = 0; edge < 12; ++edge) {
        DrawLine3D(
            ra_vector3(ra_cvert(position, rotation, edges[edge][0])),
            ra_vector3(ra_cvert(position, rotation, edges[edge][1])),
            wire_color);
    }
}

static void ra_draw(RaRenderHost* host, const RaState* state,
        const RaPose* snapshot_links) {
    static int screenshot_taken = 0;
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(1180, 760, "PufferLib - CUDA Robot Arm Manipulation");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_R)) {
        host->reset_requested = 1;
    }

    if (host->renderer == NULL) {
        host->renderer = (RaRenderer*)calloc(1, sizeof(*host->renderer));
    }
    RaRenderer* renderer = host->renderer;
    assert(renderer != NULL);
    if (!renderer->attempted) {
        renderer->attempted = 1;
        if (!FileExists(host->model_glb)) {
            fprintf(stderr, "Robot arm model not found: %s\n",
                host->model_glb);
        } else {
            renderer->arm = LoadModel(host->model_glb);
        }
        if (renderer->arm.meshCount <= 0) {
            if (FileExists(host->model_glb)) {
                fprintf(stderr, "Robot arm GLB failed to load: %s\n",
                    host->model_glb);
            }
        } else {
            renderer->cube = LoadModelFromMesh(GenMeshCube(
                2.0f*RA_CUBE_HALF, 2.0f*RA_CUBE_HALF,
                2.0f*RA_CUBE_HALF));
            renderer->cube_loaded = renderer->cube.meshCount > 0;
            const char* lighting_vs =
                "resources/robot_arm/panda_lighting.vs";
            const char* lighting_fs =
                "resources/robot_arm/panda_lighting.fs";
            if (FileExists(lighting_vs) && FileExists(lighting_fs)) {
                renderer->skin_shader = LoadShader(
                    lighting_vs, lighting_fs);
            }
            if (IsShaderValid(renderer->skin_shader)) {
                renderer->skin_shader_loaded = 1;
                renderer->light_direction_loc = GetShaderLocation(
                    renderer->skin_shader, "lightDirection");
                renderer->view_position_loc = GetShaderLocation(
                    renderer->skin_shader, "viewPosition");
                Vector3 light_direction = (Vector3){-0.42f, 0.82f, -0.38f};
                SetShaderValue(renderer->skin_shader,
                    renderer->light_direction_loc, &light_direction,
                    SHADER_UNIFORM_VEC3);
                for (int material = 0;
                        material < renderer->arm.materialCount;
                        ++material) {
                    renderer->arm.materials[material].shader =
                        renderer->skin_shader;
                }
            }
            float home[RA_DOF];
            for (int joint = 0; joint < RA_DOF; ++joint) {
                home[joint] = ra_jhome(joint);
            }
            RaPose bind[RA_LINKS];
            ra_fk(home, 0.08f, bind, NULL, NULL, NULL);
            for (int link = 0; link < RA_LINKS; ++link) {
                renderer->inverse_bind[link] =
                    MatrixInvert(ra_matrix(bind[link]));
            }
            renderer->loaded = 1;
            if (renderer->arm.meshCount != RA_EXPECTED_MESHES) {
                fprintf(stderr,
                    "Robot arm GLB has %d meshes; expected %d. "
                    "Rendering without articulated group map.\n",
                    renderer->arm.meshCount, RA_EXPECTED_MESHES);
            }
        }
    }

    if (!host->camera_initialized) {
        host->camera.target = state->basketball_mode
            ? (Vector3){0.75f, 0.38f, -0.16f}
            : (Vector3){0.23f, 0.29f, 0.0f};
        host->camera.up = (Vector3){0, 1, 0};
        host->camera.fovy = 42.0f;
        host->camera.projection = CAMERA_PERSPECTIVE;
        host->camera_initialized = 1;
    }
    float dt = GetFrameTime();
    Vector2 mouse_delta = GetMouseDelta();
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        host->camera_yaw += mouse_delta.x * 0.006f;
        host->camera_pitch += mouse_delta.y * 0.006f;
    }
    if (IsKeyDown(KEY_LEFT)) {
        host->camera_yaw -= 0.9f * dt;
    }
    if (IsKeyDown(KEY_RIGHT)) {
        host->camera_yaw += 0.9f * dt;
    }
    if (IsKeyDown(KEY_UP)) {
        host->camera_pitch += 0.65f * dt;
    }
    if (IsKeyDown(KEY_DOWN)) {
        host->camera_pitch -= 0.65f * dt;
    }
    host->camera_pitch = ra_clamp(host->camera_pitch, 0.16f, 1.15f);
    if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)
            || IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
        float sin_yaw = sinf(host->camera_yaw);
        float cos_yaw = cosf(host->camera_yaw);
        float sin_pitch = sinf(host->camera_pitch);
        float cos_pitch = cosf(host->camera_pitch);
        Vector3 right = (Vector3){sin_yaw, 0.0f, -cos_yaw};
        Vector3 up = (Vector3){
            -cos_yaw*sin_pitch, cos_pitch, -sin_yaw*sin_pitch};
        float pan_scale = 0.001f * host->camera_distance;
        host->camera.target = Vector3Add(host->camera.target,
            Vector3Add(Vector3Scale(right, -mouse_delta.x*pan_scale),
                Vector3Scale(up, mouse_delta.y*pan_scale)));
    }
    host->camera_distance = ra_clamp(
        host->camera_distance - 0.12f * GetMouseWheelMove(), 0.85f, 3.2f);
    if (IsKeyPressed(KEY_HOME)) {
        host->camera.target = state->basketball_mode
            ? (Vector3){0.75f, 0.38f, -0.16f}
            : (Vector3){0.23f, 0.29f, 0.0f};
        host->camera_distance = state->basketball_mode ? 2.35f : 1.55f;
        host->camera_yaw = 0.78f;
        host->camera_pitch = 0.48f;
    }
    float horizontal = host->camera_distance * cosf(host->camera_pitch);
    host->camera.position = (Vector3){
        host->camera.target.x + horizontal*cosf(host->camera_yaw),
        host->camera.target.y
            + host->camera_distance*sinf(host->camera_pitch),
        host->camera.target.z + horizontal*sinf(host->camera_yaw),
    };

    if (renderer->skin_shader_loaded) {
        SetShaderValue(renderer->skin_shader,
            renderer->view_position_loc, &host->camera.position,
            SHADER_UNIFORM_VEC3);
    }

    RaPose links[RA_LINKS];
    RaVec3 end_effector;
    if (snapshot_links != NULL) {
        memcpy(links, snapshot_links, sizeof(links));
        end_effector = state->end_effector;
    } else {
        ra_fk(state->q, state->gripper_width, links, NULL, NULL,
            &end_effector);
    }

    BeginDrawing();
    ClearBackground((Color){18, 23, 29, 255});
    BeginMode3D(host->camera);
    DrawCubeV(
        (Vector3){RA_TABLE_CENTER_X,
            RA_TABLE_TOP - 0.5f*RA_TABLE_THICKNESS, 0},
        (Vector3){RA_TABLE_SIZE_X, RA_TABLE_THICKNESS, RA_TABLE_SIZE_Z},
        (Color){54, 61, 68, 255});
    if (state->basketball_mode) {
        Vector3 board = {
            RA_HOOP_CENTER_X, RA_BACKBOARD_CENTER_Y,
            RA_BACKBOARD_CENTER_Z};
        DrawCubeV(board,
            (Vector3){2.0f*RA_BACKBOARD_HALF_X,
                2.0f*RA_BACKBOARD_HALF_Y, 2.0f*RA_BACKBOARD_HALF_Z},
            (Color){226, 232, 238, 255});
        DrawCubeWiresV(board,
            (Vector3){2.0f*RA_BACKBOARD_HALF_X,
                2.0f*RA_BACKBOARD_HALF_Y, 2.0f*RA_BACKBOARD_HALF_Z},
            (Color){90, 102, 116, 255});
        DrawCylinderEx(
            (Vector3){RA_HOOP_CENTER_X, RA_TABLE_TOP,
                RA_BACKBOARD_CENTER_Z - 0.025f},
            (Vector3){RA_HOOP_CENTER_X, RA_BACKBOARD_CENTER_Y,
                RA_BACKBOARD_CENTER_Z - 0.025f},
            0.012f, 0.012f, 12, (Color){72, 82, 94, 255});
        for (int segment = 0; segment < 32; ++segment) {
            float angle_a = 6.283185307f * (float)segment / 32.0f;
            float angle_b = 6.283185307f * (float)(segment + 1) / 32.0f;
            Vector3 rim_a = {
                RA_HOOP_CENTER_X + RA_RIM_MAJOR_RADIUS*cosf(angle_a),
                RA_HOOP_CENTER_Y,
                RA_HOOP_CENTER_Z + RA_RIM_MAJOR_RADIUS*sinf(angle_a)};
            Vector3 rim_b = {
                RA_HOOP_CENTER_X + RA_RIM_MAJOR_RADIUS*cosf(angle_b),
                RA_HOOP_CENTER_Y,
                RA_HOOP_CENTER_Z + RA_RIM_MAJOR_RADIUS*sinf(angle_b)};
            DrawCylinderEx(rim_a, rim_b, RA_RIM_TUBE_RADIUS,
                RA_RIM_TUBE_RADIUS, 8, (Color){235, 91, 31, 255});
        }
        for (int strand = 0; strand < 12; ++strand) {
            float angle = 6.283185307f * (float)strand / 12.0f;
            Vector3 top = {
                RA_HOOP_CENTER_X + RA_RIM_MAJOR_RADIUS*cosf(angle),
                RA_HOOP_CENTER_Y,
                RA_HOOP_CENTER_Z + RA_RIM_MAJOR_RADIUS*sinf(angle)};
            Vector3 bottom = {
                RA_HOOP_CENTER_X + 0.038f*cosf(angle + 0.20f),
                RA_HOOP_CENTER_Y - 0.11f,
                RA_HOOP_CENTER_Z + 0.038f*sinf(angle + 0.20f)};
            DrawCylinderEx(top, bottom, 0.0008f, 0.0008f, 5,
                (Color){235, 235, 225, 190});
        }
        float quality = ra_btq(
            state->cube_position, state->cube_velocity);
        unsigned char red = (unsigned char)(235.0f - 175.0f*quality);
        unsigned char green = (unsigned char)(70.0f + 175.0f*quality);
        Color path_color = (Color){red, green, 55, 255};
        RaVec3 position = state->cube_position;
        RaVec3 velocity = state->cube_velocity;
        RaVec3 visual_apex = position;
        for (int frame = 0; frame < 120; ++frame) {
            RaVec3 previous = position;
            for (int substep = 0; substep < RA_SUBSTEPS; ++substep) {
                velocity = ra_bvel(
                    velocity, RA_PHYSICS_DT);
                position = ra_add(position,
                    ra_scale(velocity, RA_PHYSICS_DT));
            }
            if (position.y > visual_apex.y) {
                visual_apex = position;
            }
            DrawLine3D(
                ra_vector3(previous), ra_vector3(position), path_color);
            if (position.y <= RA_TABLE_TOP + RA_BALL_RADIUS && frame > 1) {
                break;
            }
        }
        DrawSphere(ra_vector3(visual_apex), 0.012f,
            (Color){255, 215, 70, 255});

        RaVec3 crossing;
        if (ra_bxing(
                state->cube_position, state->cube_velocity,
                &crossing, NULL, NULL)) {
            RaVec3 hoop = ra_hoop();
            DrawSphere(ra_vector3(crossing), 0.016f, path_color);
            DrawLine3D(
                ra_vector3(crossing), ra_vector3(hoop), path_color);
        }
    } else if (state->stack_mode) {
        ra_drawc(renderer, state->base_cube_position,
            state->base_cube_rotation, (Color){235, 82, 82, 255},
            (Color){255, 225, 225, 255});
    } else {
        Vector3 target = ra_vector3(state->target_position);
        DrawCylinderEx(
            (Vector3){target.x, RA_TABLE_TOP + 0.002f, target.z},
            (Vector3){target.x, RA_TABLE_TOP + 0.012f, target.z},
            0.066f, 0.066f, 32, (Color){31, 205, 150, 150});
        DrawCylinderWiresEx(
            (Vector3){target.x, RA_TABLE_TOP + 0.013f, target.z},
            (Vector3){target.x, RA_TABLE_TOP + 0.018f, target.z},
            0.054f, 0.054f, 32, (Color){84, 255, 190, 255});
    }
    if (renderer->loaded) {
        static const signed char mesh_link[RA_EXPECTED_MESHES] = {
            -1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9,
        };
        int articulated = renderer->arm.meshCount == RA_EXPECTED_MESHES;
        for (int mesh = 0; mesh < renderer->arm.meshCount; ++mesh) {
            int link = articulated ? mesh_link[mesh] : -1;
            Matrix transform = link >= 0
                ? MatrixMultiply(
                    renderer->inverse_bind[link],
                    ra_matrix(links[link]))
                : MatrixIdentity();
            int material_index = renderer->arm.meshMaterial[mesh];
            if (material_index < 0
                    || material_index >= renderer->arm.materialCount) {
                material_index = 0;
            }
            DrawMesh(renderer->arm.meshes[mesh],
                renderer->arm.materials[material_index], transform);
        }
    }
    if (state->basketball_mode) {
        Color ball_color = state->grasped
            ? (Color){255, 190, 56, 255} : (Color){225, 112, 31, 255};
        DrawSphereEx(ra_vector3(state->cube_position), RA_BALL_RADIUS,
            12, 18, ball_color);
    } else {
        Color cube_color = state->grasped
            ? (Color){255, 196, 55, 255} : (Color){70, 155, 255, 255};
        ra_drawc(renderer, state->cube_position,
            state->cube_rotation, cube_color,
            (Color){230, 242, 255, 255});
    }
    DrawSphere(ra_vector3(end_effector), 0.012f,
        (Color){255, 218, 80, 220});
    EndMode3D();
    if (state->basketball_mode) {
        DrawText(TextFormat("Baskets: %d", state->baskets),
            24, 22, 28, (Color){245, 245, 240, 255});
    }
    EndDrawing();

    const char* screenshot = getenv("PUFFER_ROBOT_ARM_SCREENSHOT");
    if (!screenshot_taken && screenshot != NULL && screenshot[0] != '\0') {
        TakeScreenshot(screenshot);
        screenshot_taken = 1;
    }
}

static void ra_rclose(RaRenderHost* host) {
    if (host->renderer != NULL) {
        if (host->renderer->loaded) {
            UnloadModel(host->renderer->arm);
        }
        if (host->renderer->cube_loaded) {
            UnloadModel(host->renderer->cube);
        }
        if (host->renderer->skin_shader_loaded) {
            UnloadShader(host->renderer->skin_shader);
        }
        free(host->renderer);
        host->renderer = NULL;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}
