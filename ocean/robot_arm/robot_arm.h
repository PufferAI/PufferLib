#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef RA_OBS_T_DEFINED
#define RA_OBS_T_DEFINED
#if defined(from_float) && !defined(PRECISION_FLOAT)
typedef precision_t obs_t;
#else
typedef float obs_t;
#endif
#endif
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
#define RA_ROBOT_FRICTION 0.80f
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
    RaVec3 previous_cube_position;
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

RA_HD static RA_INLINE RaVec3 ra_cvert(
        RaVec3 position, RaQuat rotation, int vertex) {
    RaVec3 local = ra_v3(
        (vertex & 1) ? RA_CUBE_HALF : -RA_CUBE_HALF,
        (vertex & 2) ? RA_CUBE_HALF : -RA_CUBE_HALF,
        (vertex & 4) ? RA_CUBE_HALF : -RA_CUBE_HALF);
    return ra_add(position, ra_rotate(rotation, local));
}

RA_HD static RA_INLINE float ra_jhome(int joint) {
    const float values[RA_DOF] = {0.00f, -0.785398f, 0.00f, -2.356194f,
        0.00f, 1.570796f, 0.785398f};
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

#define RA_EXPECTED_MESHES 11

typedef struct RaRenderer {
    Model arm;
    Model cube;
    Shader skin_shader;
    Matrix inverse_bind[RA_LINKS];
    int light_direction_loc;
    int view_position_loc;
    int loaded;
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
    DrawModelEx(renderer->cube, ra_vector3(position), axis,
        angle * RAD2DEG, (Vector3){1, 1, 1}, color);
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

    RaRenderer* renderer = host->renderer;
    assert(renderer != NULL);
    if (!renderer->loaded) {
        assert(FileExists(host->model_glb));
        renderer->arm = LoadModel(host->model_glb);
        assert(renderer->arm.meshCount == RA_EXPECTED_MESHES);
        renderer->cube = LoadModelFromMesh(GenMeshCube(
            2.0f*RA_CUBE_HALF, 2.0f*RA_CUBE_HALF,
            2.0f*RA_CUBE_HALF));
        assert(renderer->cube.meshCount > 0);
        const char* lighting_vs = "resources/robot_arm/panda_lighting.vs";
        const char* lighting_fs = "resources/robot_arm/panda_lighting.fs";
        assert(FileExists(lighting_vs) && FileExists(lighting_fs));
        renderer->skin_shader = LoadShader(lighting_vs, lighting_fs);
        assert(IsShaderValid(renderer->skin_shader));
        renderer->light_direction_loc = GetShaderLocation(
            renderer->skin_shader, "lightDirection");
        renderer->view_position_loc = GetShaderLocation(
            renderer->skin_shader, "viewPosition");
        Vector3 light_direction = (Vector3){-0.42f, 0.82f, -0.38f};
        SetShaderValue(renderer->skin_shader,
            renderer->light_direction_loc, &light_direction,
            SHADER_UNIFORM_VEC3);
        for (int material = 0; material < renderer->arm.materialCount;
                ++material) {
            renderer->arm.materials[material].shader =
                renderer->skin_shader;
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

    SetShaderValue(renderer->skin_shader,
        renderer->view_position_loc, &host->camera.position,
        SHADER_UNIFORM_VEC3);

    const RaPose* links = snapshot_links;
    RaVec3 end_effector = state->end_effector;

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
    static const signed char mesh_link[RA_EXPECTED_MESHES] = {
        -1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9,
    };
    for (int mesh = 0; mesh < RA_EXPECTED_MESHES; ++mesh) {
        int link = mesh_link[mesh];
        Matrix transform = link >= 0
            ? MatrixMultiply(
                renderer->inverse_bind[link],
                ra_matrix(links[link]))
            : MatrixIdentity();
        DrawMesh(renderer->arm.meshes[mesh],
            renderer->arm.materials[renderer->arm.meshMaterial[mesh]],
            transform);
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
            UnloadModel(host->renderer->cube);
            UnloadShader(host->renderer->skin_shader);
        }
        free(host->renderer);
        host->renderer = NULL;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}
