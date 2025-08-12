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

// === KINEMATICS: Workspace bounds (meters) ===
// Defines the reachable Cartesian volume for the end-effector.
// Used for target sampling and workspace clamping.
#define WORKSPACE_X_MIN -0.5f
#define WORKSPACE_X_MAX  0.5f
#define WORKSPACE_Y_MIN -0.5f
#define WORKSPACE_Y_MAX  0.5f
#define WORKSPACE_Z_MIN  0.0f
#define WORKSPACE_Z_MAX  0.8f

// === KINEMATICS: Link lengths (meters) ===
// Geometric parameters for forward kinematics.
#define ARM_LINK1_LENGTH 0.35f
#define ARM_LINK2_LENGTH 0.25f
#define ARM_LINK3_LENGTH 0.10f

// Pick-and-place objects
#define MAX_OBJECTS 4
#define MAX_BASKETS 3
#define OBJECT_SIZE 0.03f  // 3cm cubes
#define BASKET_SIZE 0.08f  // 8cm basket diameter


// Time step (seconds)
#define ARM_DT 0.02f

// === PHYSICS: World and contact parameters ===
// Gravity, table height, object material and contact properties.
#define GRAVITY -9.81f           // m/s^2
#define TABLE_HEIGHT 0.25f       // Table surface height
#define OBJECT_MASS 0.02f        // 20g objects (lighter for easier manipulation)
#define OBJECT_RESTITUTION 0.2f  // Reduced bounce for stability
#define OBJECT_FRICTION 0.8f     // Higher friction to prevent sliding
#define AIR_DAMPING 0.95f        // Stronger air resistance for stability
#define CONTACT_STIFFNESS 300.0f // Softer contact for gentler physics
#define CONTACT_DAMPING 30.0f    // Reduced damping for responsiveness

// === PHYSICS: Gripper contact model ===
// Approximates fingertip contact and force limits.
#define GRIPPER_FINGER_LENGTH 0.05f  // Slightly longer fingers for better reach
#define GRIPPER_MAX_FORCE 2.0f       // Softer max force to avoid launching objects
#define GRIPPER_CONTACT_RADIUS 0.035f // Larger contact radius for easier grasping


// Pick-and-place object types
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

// === PHYSICS: Rigid body state ===
// Minimal translational/rotational state with force/torque accumulators.
typedef struct {
    float pos[3];           // world position
    float vel[3];           // linear velocity (m/s)
    float angular_vel[3];   // angular velocity (rad/s)
    float orientation[4];   // quaternion rotation (w,x,y,z)
    float force[3];         // accumulated forces this frame
    float torque[3];        // accumulated torques this frame
    bool on_surface;        // resting on table/ground
    bool in_contact;        // in contact with gripper
    float contact_time;     // time in contact (for grip stability)
} PhysicsBody;

// === PHYSICS: Manipulated object with material properties ===
typedef struct {
    PhysicsBody physics;    // rigid body physics state
    ObjectType type;        // color type
    bool grasped;           // currently grasped by robot
    bool in_basket;         // placed in correct basket
    int target_basket;      // which basket this object should go to
    
    // Bounding box for collision detection
    float size[3];          // object dimensions (x,y,z)
    float mass;             // object mass
    float restitution;      // bounce factor
    float friction;         // surface friction
} ManipObject;

// === PHYSICS: Gripper finger contact surrogate ===
typedef struct {
    float pos[3];           // finger tip position
    float vel[3];           // finger velocity
    float force[3];         // contact force being applied
    bool in_contact;        // touching an object
    int contact_object_id;  // which object is being touched (-1 if none)
} GripperFinger;

// Collection basket
typedef struct {
    float pos[3];           // world position  
    BasketType type;        // basket color
    float collected_count;  // how many objects collected
} Basket;

// Required Log layout (floats only)
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
    float pick_success_rate;    // successful picks per episode
    float place_success_rate;   // successful placements per episode
} Log;

// Enhanced robot arm with pick-and-place capabilities
typedef struct RobotArm {
    // Required by env_binding.h
    Log   log;
    float *observations;   // shape (19,) enhanced for pick-and-place, (14,) for reach
    float *actions;        // shape (7,) joint velocities + gripper
    float *rewards;        // shape (1,)
    unsigned char *terminals; // shape (1,)

    // Episode control
    int max_steps;
    int episode_steps;

    // Mode control
    int task;                 // 0=reach, 1=pick_and_place
    int reach_only;           // if true force simple reach task
    int pick_and_place_mode;  // enable full pick-and-place functionality

    // === KINEMATICS: Joint angles and end-effector pose ===
    float joint_angles[6];        // base, shoulder, elbow, wrist_roll, wrist_pitch, wrist_yaw
    float end_effector[3];        // x,y,z
    float end_effector_orient[3]; // roll,pitch,yaw (approx)
    
    // === PHYSICS: Gripper contact state ===
    float gripper_state;          // 0.0=open, 1.0=closed
    float gripper_command;        // target gripper state from actions
    float gripper_force;          // current gripping force (N)
    GripperFinger left_finger;    // left gripper finger
    GripperFinger right_finger;   // right gripper finger
    int grasped_object_id;        // ID of currently grasped object (-1 if none)
    float grasp_stability;        // how stable the grasp is (0-1)
    int   grasp_event;            // set to 1 when a new grasp occurs this step

    // Pick-and-place environment
    ManipObject objects[MAX_OBJECTS];  // objects to manipulate
    Basket baskets[MAX_BASKETS];       // collection baskets
    int current_target_object;         // which object to pick next
    ObjectType target_type;            // currently targeted color/type
    int task_stage;                    // 0=reach, 1=grasp, 2=lift, 3=transport, 4=place

    // Simple reach mode (for backwards compatibility)
    float target_pos[3];
    float target_orient[3];

    // Caches
    float cached_sin[6];
    float cached_cos[6];
    float prev_distance;
    float current_distance;

    float episode_score_accum;
    float episode_return_accum;

    // === DYNAMICS: Joint velocity and command filtering ===
    float joint_vel[6];
    float cmd_filt[6];

    // Camera
    Camera3D camera;
    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
    bool camera_initialized;

    // Training knobs
    int   frame_skip;
    float obs_noise_std;
    float actuation_noise_std;
    float success_distance;
    int   domain_randomization;

    // Motion smoothing knobs
    float action_smoothing_alpha;
    float accel_limit;
    float damping;

    // === KINEMATICS: Link parameters (support domain randomization) ===
    float link1_length;
    float link2_length;
    float link3_length;

    // Curriculum and spawning controls (reach mode)
    int   episodes_completed;
    int   curriculum_episodes;
    float success_distance_start;
    float success_distance_min;
    float on_gripper_spawn_start;
    float on_gripper_spawn_min;
    float on_gripper_spawn_prob;
    float start_grasp_prob;

    // Per-target timing for speed bonuses
    int   target_spawn_step;
    int   target_touch_awarded;
    float touch_bonus_max;
    int   touch_decay_steps;

    // Anti-stall tracking
    int   stagnation_steps;
    int   stagnation_limit;
    float best_metric;

    int   assist_enabled;
    int   assist_episodes;
} RobotArm;

// Required functions
void c_reset(RobotArm *env);
void c_step(RobotArm *env);
void c_render(RobotArm *env);
void c_close(RobotArm *env);
#endif // PUFFERLIB_OCEAN_ROBOT_ARM_H
