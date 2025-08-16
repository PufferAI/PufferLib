#ifndef PUFFERLIB_OCEAN_ROBOT_ARM_H
#define PUFFERLIB_OCEAN_ROBOT_ARM_H
#include <stdbool.h>
#include "raylib.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923f
#endif
#define WORKSPACE_X_MIN -0.65f
#define WORKSPACE_X_MAX  0.65f
#define WORKSPACE_Y_MIN -0.65f
#define WORKSPACE_Y_MAX  0.65f
#define WORKSPACE_Z_MIN  0.0f
#define WORKSPACE_Z_MAX  0.8f
#define ARM_LINK1_LENGTH 0.35f
#define ARM_LINK2_LENGTH 0.25f
#define ARM_LINK3_LENGTH 0.10f
#define MAX_OBJECTS 4
#define MAX_BASKETS 3
#define OBJECT_SIZE 0.03f
#define BASKET_SIZE 0.08f
#define ARM_DT 0.01f
#define GRAVITY -9.81f
#define TABLE_HEIGHT 0.25f
#define OBJECT_MASS 0.05f
#define OBJECT_RESTITUTION 0.1f
#define OBJECT_FRICTION 0.8f
#define AIR_DAMPING 0.90f
#define CONTACT_STIFFNESS 300.0f
#define CONTACT_DAMPING 30.0f
#define GRIPPER_FINGER_LENGTH 0.05f
#define GRIPPER_MAX_FORCE 2.0f
#define GRIPPER_CONTACT_RADIUS 0.060f
#ifndef ARM_USE_EULER
#define ARM_USE_EULER 1
#endif


typedef enum {
    OBJ_RED = 0,
    OBJ_BLUE = 1,
    OBJ_GREEN = 2,
    OBJ_YELLOW = 3
} ObjectType;

typedef enum {
    BASKET_RED = 0,
    BASKET_BLUE = 1,
    BASKET_GREEN = 2
} BasketType;

typedef struct {
    float pos[3];
    float vel[3];
    float angular_vel[3];
    float orientation[4];
    float force[3];
    float torque[3];
    bool on_surface;
    bool in_contact;
    float contact_time;
} PhysicsBody;


typedef struct {
    PhysicsBody physics;
    ObjectType type;
    bool grasped;
    bool in_basket;
    int target_basket;
    float size[3];
    float mass;
    float restitution;
    float friction;
} ManipObject;


typedef struct {
    float pos[3];
    float vel[3];
    float force[3];
    bool in_contact;
    int contact_object_id;
} GripperFinger;

typedef struct {
    float pos[3];
    BasketType type;
    float collected_count;
} Basket;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float pick_success_rate;
    float place_success_rate;
} Log;


typedef struct RobotArm {
    Log   log;
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
    int max_steps;
    int episode_steps;
    int task;
    int pick_and_place_mode;
    float joint_angles[6];
    float end_effector[3];
    float end_effector_orient[3];
    float gripper_state;
    float gripper_command;
    float gripper_force;
    GripperFinger left_finger;
    GripperFinger right_finger;
    int grasped_object_id;
    float grasp_stability;
    int   grasp_event;
    int   was_grasped_last_step;
    ManipObject objects[MAX_OBJECTS];
    Basket baskets[MAX_BASKETS];
    int   basket_disabled[MAX_BASKETS];
    int current_target_object;
    ObjectType target_type;
    int task_stage;
    float target_pos[3];
    float target_orient[3];
    float cached_sin[6];
    float cached_cos[6];
    float prev_distance;
    float current_distance;
    float best_place_dist;
    float episode_score_accum;
    float episode_return_accum;
    float episode_pick_count;
    float episode_place_count;
    float joint_vel[6];
    float cmd_filt[6];
    Camera3D camera;
    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
    bool camera_initialized;
    int   frame_skip;
    float obs_noise_std;
    float actuation_noise_std;
    float success_distance;
    float success_distance2;
    int   domain_randomization;
    float action_penalty_coef;
    float reward_scale;
    int   headless;
    int   render_decimation;
    int   render_counter;
    int   render_target_fps;
    int   vsync;
    Model cube_model;
    int   cube_model_loaded;
    float cube_model_scale;
    float cube_model_offset[3];
    float cube_model_visual_mul;
    float action_smoothing_alpha;
    float accel_limit;
    float damping;
    float link1_length;
    float link2_length;
    float link3_length;
    int   episodes_completed;
    int   curriculum_episodes;
    float success_distance_start;
    float success_distance_min;
    float on_gripper_spawn_start;
    float on_gripper_spawn_min;
    float on_gripper_spawn_prob;
    float start_grasp_prob;
    int   target_spawn_step;
    int   target_touch_awarded;
    float touch_bonus_max;
    int   touch_decay_steps;
    int   stagnation_steps;
    int   stagnation_limit;
    float best_metric;
    int   assist_enabled;
    int   assist_episodes;
    int   continuous_gripper;
    int   extended_observation;
    int   terminate_on_place;
    int   use_unified_clamp;
    float unified_clamp_min;
    float unified_clamp_max;
    int   placed_event;
    int   oob_steps;
    int   early_reach_episodes;
    float early_reach_bonus;
    int   near_object_steps;
    int   target_unreachable_steps;
    int   nearest_target_idx;
    float nearest_target_d2;
    int   fk_dirty;
    int   physics_substeps;
    int   release_event;
    int   recent_release_object_id;
} RobotArm;

void c_reset(RobotArm *env);
void c_step(RobotArm *env);
void c_render(RobotArm *env);
void c_close(RobotArm *env);
#endif
