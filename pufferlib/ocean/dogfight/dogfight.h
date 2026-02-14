// dogfight.h - WW2 aerial combat environment
// Uses flightlib.h for flight physics

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "rlgl.h"  // For rlSetClipPlanes()

#define DEBUG 0
#define EVAL_WINDOW 50
#define PENALTY_STALL 0.002f
#define PENALTY_RUDDER 0.001f

#include "flightlib.h"
#include "autopilot.h"
#include "autoace.h"

typedef enum {
    OBS_MOMENTUM_GFORCE = 0,    // G-force awareness (17 obs) — proven winner from df24
    OBS_DRONE_STYLE = 1,        // + quaternion + up vector (23 obs)
    OBS_PILOT_QUAT = 2,         // Pilot + quaternion (26 obs)
    OBS_PILOT = 3,              // Pilot awareness (22 obs)
    OBS_SCHEME_COUNT
} ObsScheme;

static const int OBS_SIZES[OBS_SCHEME_COUNT] = {17, 23, 26, 22};

typedef enum {
    CURRICULUM_TAIL_CHASE = 0,       // Stage 0: Easiest - opponent ahead, same heading
    CURRICULUM_HEAD_ON,              // Stage 1: Opponent coming toward us
    CURRICULUM_VERTICAL,             // Stage 2: Above or below player
    CURRICULUM_GENTLE_TURNS,         // Stage 3: Opponent does gentle 30° turns
    CURRICULUM_OFFSET,               // Stage 4: Large lateral/vertical offset, same heading
    CURRICULUM_ANGLED,               // Stage 5: Offset + different heading (±22°)
    CURRICULUM_SIDE_NEAR,            // Stage 6: 15-45° off axis (NEW - small side turn)
    CURRICULUM_SIDE_MID,             // Stage 7: 30-60° off axis (NEW - medium side turn)
    CURRICULUM_SIDE_FAR,             // Stage 8: 45-90° off axis (was SIDE_CHASE)
    CURRICULUM_SIDE_MANEUVERING,     // Stage 9: Side chase + 30° turns
    CURRICULUM_DIVE_ATTACK,          // Stage 10: 500m altitude advantage, 75° nose-down dive
    CURRICULUM_ZOOM_ATTACK,          // Stage 11: 500m below, 75° nose-up, near max speed
    CURRICULUM_REAR_CHASE,           // Stage 12: Target 90-150° off axis (rear quarters)
    CURRICULUM_REAR_MANEUVERING,     // Stage 13: Rear chase + 30° turns
    CURRICULUM_FULL_PREDICTABLE,     // Stage 14: 360° spawn, heading correlated (flying away)
    CURRICULUM_FULL_RANDOM,          // Stage 15: 360° spawn, random heading, 30° turns
    CURRICULUM_MEDIUM_TURNS,         // Stage 16: 360° spawn, random heading, 45° turns
    CURRICULUM_HARD_MANEUVERING,     // Stage 17: 60° turns + weave patterns
    CURRICULUM_CROSSING,             // Stage 18: 45 degree deflection shots
    CURRICULUM_EVASIVE,              // Stage 19: Reactive evasion (hardest)
    CURRICULUM_AUTOACE,              // Stage 20: Full AutoAce opponent (two-way combat)
    CURRICULUM_COUNT                 // = 21
} CurriculumStage;

// Forward declarations for stage spawn functions
struct Dogfight;  // Forward declare Dogfight struct
typedef void (*SpawnFn)(struct Dogfight*, Vec3, Vec3);

// Stage configuration: consolidates all stage metadata in one place
typedef struct StageConfig {
    int n;                      // Stage number (for easy lookup)
    SpawnFn spawn;              // Function pointer to spawn function
    const char* description;    // Human-readable description
    float weight;               // Difficulty weight (0.0-1.0)
    int max_steps;              // Episode length - fail to kill = terminal + score -1
    float angle_min_deg;        // Min angle off axis (for documentation)
    float angle_max_deg;        // Max angle off axis
    int bank;                   // Target bank angle in degrees (0=straight, 30, 45, 60)
} StageConfig;

// Forward declarations of spawn functions (defined below, after Dogfight struct)
static void spawn_tail_chase(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_head_on(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_vertical(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_gentle_turns(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_offset(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_angled(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_side(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_dive_attack(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_zoom_attack(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_rear(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_full_predictable(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_full_random(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_medium_turns(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_hard_maneuvering(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_crossing(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_evasive(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_autoace(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);

// Stage configuration table - single source of truth for all stage metadata
// Updated 2026-01-25 to split SIDE_CHASE into 3 stages (SIDE_NEAR, SIDE_MID, SIDE_FAR)
// Updated 2026-01-26: Consolidated spawn_side/spawn_rear functions use angle fields
// Updated 2026-01-27: Added DIVE_ATTACK (10) and ZOOM_ATTACK (11) stages
// max_steps field is now for documentation only; episode length comes from Python config
static const StageConfig STAGES[CURRICULUM_COUNT] = {
    // n   spawn_fn               description                          weight  max_steps  ang_min ang_max bank
    {0,  spawn_tail_chase,       "Target ahead, same heading",         0.01f,  300,       0,      10,     0},
    {1,  spawn_head_on,          "Target coming toward us",            0.02f,  300,       170,    180,    0},
    {2,  spawn_vertical,         "Target above/below",                 0.05f,  500,       0,      20,     0},
    {3,  spawn_gentle_turns,     "Target ahead, 30 deg turns",         0.10f,  1000,      0,      30,     30},
    {4,  spawn_offset,           "Large lateral offset, 30 deg turns", 0.15f,  1000,      0,      45,     30},
    {5,  spawn_angled,           "Offset + heading variance, 30 deg",  0.20f,  1200,      0,      22,     30},
    {6,  spawn_side,             "15-45 deg off axis",                 0.25f,  1500,      15,     45,     0},
    {7,  spawn_side,             "30-60 deg off axis",                 0.30f,  1800,      30,     60,     0},
    {8,  spawn_side,             "45-90 deg off axis",                 0.35f,  2000,      45,     90,     0},
    {9,  spawn_side,             "45-90 deg + 30 deg turns",           0.40f,  3000,      45,     90,     30},
    {10, spawn_dive_attack,      "Dive attack, 500m altitude adv",     0.45f,  2500,      120,     175,    0},
    {11, spawn_zoom_attack,      "Zoom attack, 75 deg nose-up",        0.50f,  3000,      120,     175,    0},
    {12, spawn_rear,             "90-150 deg off axis",                0.58f,  3500,      90,     150,    0},
    {13, spawn_rear,             "90-150 deg + 30 deg turns",          0.62f,  3500,      90,     150,    30},
    {14, spawn_full_predictable, "360 deg, heading correlated",        0.68f,  4000,      0,      360,    0},
    {15, spawn_full_random,      "360 deg random heading, 30 deg",     0.74f,  4000,      0,      360,    30},
    {16, spawn_medium_turns,     "360 deg, 45 deg bank turns",         0.82f,  4000,      0,      360,    45},
    {17, spawn_hard_maneuvering, "360 deg, 60 deg banks + weave",      0.90f,  4000,      0,      360,    60},
    {18, spawn_crossing,         "45 deg deflection shots",            0.95f,  4000,      45,     45,     0},
    {19, spawn_evasive,          "Reactive break turns",               1.00f,  4000,      0,      360,    60},
    {20, spawn_autoace,          "AutoAce intelligent opponent",       1.00f,  6000,      0,      360,    0},
};

// Spawn randomization parameters - stage-dependent ranges for variety
typedef struct SpawnRandomization {
    float speed_min, speed_max;       // Initial airspeed range (m/s)
    float pitch_max_deg;              // Max pitch deviation (±degrees)
    float bank_max_deg;               // Max bank deviation (±degrees)
    float throttle_min, throttle_max; // Initial throttle range
} SpawnRandomization;

// Get spawn randomization parameters for a given stage
// Earlier stages = tighter ranges (easier), later stages = wider ranges (harder)
// Updated 2026-01-27: Stage boundaries adjusted for 20-stage curriculum (added DIVE_ATTACK, ZOOM_ATTACK)
static inline SpawnRandomization get_spawn_randomization(int stage) {
    if (stage <= 3)  return (SpawnRandomization){75, 85,  5, 10, 0.45f, 0.55f};
    if (stage <= 7)  return (SpawnRandomization){70, 95, 10, 20, 0.35f, 0.65f};
    if (stage <= 13) return (SpawnRandomization){65, 105, 15, 30, 0.30f, 0.70f};
    return (SpawnRandomization){60, 110, 15, 45, 0.25f, 0.80f};
}

#define DT 0.02f

#define WORLD_HALF_X 4000.0f
#define WORLD_HALF_Y 4000.0f
#define WORLD_MAX_Z 5000.0f
#define MAX_SPEED 250.0f

#define INV_WORLD_HALF_X 0.00025f      // 1/4000
#define INV_WORLD_HALF_Y 0.00025f      // 1/4000
#define INV_WORLD_MAX_Z  0.0002f       // 1/5000
#define INV_MAX_SPEED    0.004f        // 1/250
#define INV_PI           0.31830988618f // 1/PI
#define INV_HALF_PI      0.63661977236f // 2/PI (i.e., 1/(PI*0.5))
#define DEG_TO_RAD       0.01745329252f // PI/180

#define GUN_RANGE 500.0f       // meters
#define INV_GUN_RANGE 0.002f   // 1/500
#define GUN_CONE_ANGLE 0.087f  // ~5 degrees in radians
#define FIRE_COOLDOWN 10       // ticks (0.2 seconds at 50Hz)

typedef struct Log {
    float episode_return;
    float episode_length;
    float score;           // 1.0 on kill, 0.0 on failure
    float perf;            // Raw kills (becomes kill_rate after vec_log divides by n)
    float sp_player_kills; // Self-play only: player kills (TUI shows P:## O:##)
    float sp_opp_kills;    // Self-play only: opponent kills
    float shots_fired;
    float accuracy;
    float stage;

    // RAW SUMS - exported to Python, become correct averages after vec_log divides by n
    float total_stage_weight;       // Sum of stage weights (exported as avg_stage_weight)
    float total_abs_bias;           // Sum of |aileron_bias| (exported as avg_abs_bias)
    float stage_sum;                // Sum of stages (exported as avg_stage)
    float total_control_rate;       // Sum of per-episode mean squared deltas (exported as avg_control_rate)
    float base_stage_kills;         // Kills at int(curriculum_target) - for per-stage gating
    float base_stage_eps;           // Episodes at int(curriculum_target) - for per-stage gating

    // Death spiral diagnostics - exported to Python/wandb
    float player_ground_hits;       // Player crashed into ground
    float opponent_ground_hits;     // Opponent crashed into ground
    float recovery_triggers;        // Recovery hijacking activated
    float clean_fights;             // Episodes ending in kills or timeouts (not crashes)
    float altitude_kills;           // Kills from forcing opponent crash at safe altitude

    // PER-ENV RATIOS - for C debugging only, NOT exported (garbage after vec_log aggregation)
    float avg_stage_weight;         // = total_stage_weight / n (per-env only)
    float avg_abs_bias;             // = total_abs_bias / n (per-env only)
    float avg_stage;                // = stage_sum / n (per-env only)
    float kill_rate;                // = perf / n (per-env only - Python uses 'perf' instead)
    float ultimate;                 // = kill_rate * avg_stage_weight (per-env only)
    float ultimate2;                // = kill_rate * clean_fight_rate (per-env only)
    float n;
} Log;

typedef enum DeathReason {
    DEATH_NONE = 0,      // Episode still running
    DEATH_KILL = 1,      // Player scored a kill (success)
    DEATH_OOB = 2,       // Out of bounds
    DEATH_TIMEOUT = 3,   // Max steps reached
    DEATH_SUPERSONIC = 4 // Physics blowup
} DeathReason;

typedef struct RewardConfig {
    // Positive shaping
    float aim_scale;         // Continuous aiming reward (default 0.05)
    float closing_scale;     // +N per m/s closing (default 0.003)
    // Penalties
    float neg_g;             // -N per unit G below 0.5 (default 0.02) - enforces "pull to turn"
    float control_rate_penalty;  // Penalty for (action - prev_action)^2 (default 0, sweepable)
    // Low altitude penalty (discourages death spirals)
    float low_altitude_threshold;  // Altitude below which penalty applies (default 1500.0)
    float low_altitude_penalty;    // Penalty scale at ground level (default 0.01)
    // Thresholds
    float speed_min;         // Stall threshold (default 50.0)
    // Curriculum decay (DEPRECATED - use timestep-based decay instead)
    float aim_decay_stage;   // Stage at which aim reward reaches 0 (default 15.0) - DEPRECATED
    // Timestep-based shaping decay (anneals r_aim and r_closing during self-play)
    long shaping_decay_start;    // Start annealing at this global step (0 = disabled)
    long shaping_decay_end;      // Complete annealing at this global step
} RewardConfig;

// Calculate shaping decay multiplier based on global training step
// Returns 1.0 before decay_start, 0.0 after decay_end, linear interpolation between
static inline float calc_shaping_decay(long global_step, long decay_start, long decay_end) {
    if (decay_start <= 0 || decay_end <= decay_start) return 1.0f;  // Disabled
    if (global_step < decay_start) return 1.0f;  // Before window
    if (global_step >= decay_end) return 0.0f;   // After window
    return 1.0f - (float)(global_step - decay_start) / (float)(decay_end - decay_start);
}

typedef struct Client {
    Camera3D camera;
    float width;
    float height;

    float cam_distance;
    float cam_azimuth;
    float cam_elevation;
    int camera_mode;  // 0 = follow target, 1 = midpoint view
    bool is_dragging;
    float last_mouse_x;
    float last_mouse_y;

    Model plane_model;
    Texture2D plane_texture;
    bool model_loaded;

    float propeller_angle;  // Current propeller rotation (radians)

    float last_cam_hx;  // Cached camera heading X (stable through vertical)
    float last_cam_hy;  // Cached camera heading Y (stable through vertical)
} Client;

typedef struct Dogfight {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    // Opponent perspective buffers (for dual self-play with Multiprocessing)
    // Written during c_step() if non-NULL, same size as observations/rewards
    float *opponent_observations;  // Opponent's view of the world
    float *opponent_rewards;       // = -player_reward (zero-sum)

    Log log;
    Client *client;
    int tick;
    int max_steps;
    float episode_return;
    Plane player;
    Plane opponent;
    // Per-episode precomputed values (for curriculum learning)
    float gun_cone_angle;   // Hit detection cone (radians) - FIXED at 5°
    float cos_gun_cone;     // cosf(gun_cone_angle) - for hit detection
    // Opponent autopilot
    AutopilotState opponent_ap;
    // AutoAce intelligent opponent (stage 20+)
    AutoAceState opponent_ace;
    // Observation scheme
    int obs_scheme;
    int obs_size;
    // Reward configuration (sweepable)
    RewardConfig rcfg;
    // Episode-level tracking (reset each episode)
    int kill;                   // 1 if killed this episode, 0 otherwise
    int opp_kill;               // 1 if opponent killed player this episode (self-play)
    float episode_shots_fired;  // For accuracy tracking
    // Curriculum learning
    int curriculum_enabled;     // 0 = off (legacy spawning), 1 = on
    int curriculum_randomize;   // 0 = progressive (training), 1 = random stage each episode (eval)
    int total_episodes;         // Cumulative episodes (persists across resets)
    CurriculumStage stage;      // Current difficulty stage (set globally by Python)
    float curriculum_target;    // Float 0.0-15.0 for probabilistic stage assignment
    int is_initialized;         // Flag to preserve curriculum state across re-init (for Multiprocessing)
    // Anti-spinning
    float total_aileron_usage;  // Accumulated |aileron| input (for spin death)
    float aileron_bias;         // Cumulative signed aileron (for directional penalty)
    float episode_control_rate; // Sum of squared control deltas this episode
    // Episode reward accumulators (for DEBUG summaries)
    float sum_r_closing;
    float sum_r_speed;      // Stall penalty
    float sum_r_neg_g;
    float sum_r_rudder;
    float sum_r_aim;
    float sum_r_rate;       // Control rate penalty
    // Episode accumulators for new reward terms (for debug logging)
    float sum_r_altitude;
    float sum_r_time;
    float sum_r_player_energy;
    float sum_r_energy_adv;
    // Per-tick reward components (for debug logging)
    float r_closing;
    float r_aim;
    float r_neg_g;
    float r_stall;
    float r_rudder;
    float r_rate;
    float r_altitude;
    float r_time;
    float r_player_energy;
    float r_energy_adv;
    // Aiming diagnostics (reset each episode, for DEBUG output)
    float best_aim_angle;    // Best (smallest) aim angle achieved (radians)
    int ticks_in_cone;       // Ticks where aim_dot > cos_gun_cone
    float closest_dist;      // Closest approach to target (meters)
    // Flight envelope diagnostics (reset each episode, for DEBUG output)
    float max_g, min_g;           // Peak G-forces experienced
    float max_bank;               // Peak bank angle (abs, radians)
    float max_pitch;              // Peak pitch angle (abs, radians)
    float min_speed, max_speed;   // Speed envelope (m/s)
    float min_alt, max_alt;       // Altitude envelope (m)
    float sum_throttle;           // For computing mean throttle
    int trigger_pulls;            // Times trigger was pulled (>0.5)
    int prev_trigger;             // For edge detection
    DeathReason death_reason;
    DeathReason last_death_reason;  // For rendering: what ended the previous episode
    int last_winner;                // For rendering: 1=player won, -1=opponent won, 0=draw/timeout
    // Debug
    int env_num;                // Environment index (for filtering debug output)
    // Observation highlighting (for visual debugging)
    unsigned char obs_highlight[26];  // 1 = highlight this observation with red arrow (max scheme is 26 obs)
    // Last opponent actions (for Python access in tests)
    float last_opp_actions[5];  // throttle, elevator, aileron, rudder, trigger
    // Camera control
    int camera_follow_opponent;  // 0 = follow player (default), 1 = follow opponent
    // Self-play: external opponent actions override (Phase 1)
    float opponent_actions_override[5];  // [throttle, elevator, aileron, rudder, trigger]
    int use_opponent_override;           // 0 = use autopilot, 1 = use override
    // Head-on lockout: disable guns until planes pass each other (only for head-on spawns)
    int head_on_lockout;                 // 1 = guns locked until pass-through detected
    float prev_rel_dot;                  // Previous dot(rel_pos, rel_vel) for detecting pass
    // Eval spawn mode: 0 = random (default), 1 = opponent_advantage (for testing opponent kill)
    int eval_spawn_mode;
    // Previous actions for control rate penalty
    float prev_elevator;  // Previous elevator for rate penalty
    float prev_aileron;   // Previous aileron for rate penalty
    float prev_rudder;    // Previous rudder for rate penalty
    // Late-training debug logging (activated when global_step >= debug_trigger_step)
    long global_step;           // Current training step (set by Python each tick)
    long debug_trigger_step;    // Start logging when global_step >= this value
    FILE* debug_log_file;       // File handle for debug output (NULL if not logging)
    int debug_log_initialized;  // 1 if file opened

    // Opponent recovery hijacking (breaks death spiral equilibrium in self-play)
    int selfplay_active;               // 1 when in self-play mode (set by Python)
    int opponent_recovery_active;      // 1 if recovery maneuver in progress
    int opponent_recovery_tick_start;  // Tick when recovery started
    float recovery_altitude_threshold; // Trigger altitude (default 500m)
    float recovery_trigger_prob;       // Per-tick probability (default 0.1)
    float recovery_speed_threshold;    // Speed for phase 2 (default 70m/s)
    float recovery_bank_deg;           // Turn bank angle (default 60°)
    unsigned int recovery_rng_state;   // Separate RNG for recovery triggers
    int opponent_above_recovery_threshold;  // 1 if opponent was above threshold last tick (for crossing detection)

    // Guided climb hijack (teachable opponent maneuver for self-play diversity)
    // When active, Python should override opponent actions with climb control
    // This creates training data showing "climb after merge = good"
    int guided_climb_active;           // 1 = Python should override with climb actions
    int guided_climb_ticks_remaining;  // Countdown to hand back control
    float guided_climb_elevator;       // Target elevator value for 3G climb

    // Runtime-configurable flight physics (for parameter sweeps)
    FlightParams flight_params;
} Dogfight;

#include "dogfight_observations.h"

void init(Dogfight *env, int obs_scheme, RewardConfig *rcfg, int curriculum_enabled, int curriculum_randomize, int env_num) {
    env->log = (Log){0};
    env->tick = 0;
    env->env_num = env_num;
    env->episode_return = 0.0f;
    env->client = NULL;
    // Observation scheme
    env->obs_scheme = (obs_scheme >= 0 && obs_scheme < OBS_SCHEME_COUNT) ? obs_scheme : 0;
    env->obs_size = OBS_SIZES[env->obs_scheme];
    // Gun cone for HIT DETECTION - fixed at 5°
    env->gun_cone_angle = GUN_CONE_ANGLE;
    env->cos_gun_cone = cosf(env->gun_cone_angle);
    autopilot_init(&env->opponent_ap);
    autoace_init(&env->opponent_ace);
    // Reward configuration (copy from provided config)
    env->rcfg = *rcfg;
    // Episode tracking
    env->kill = 0;
    env->episode_shots_fired = 0.0f;

    env->curriculum_enabled = curriculum_enabled;
    env->curriculum_randomize = curriculum_randomize;
    if (!env->is_initialized) {
        env->total_episodes = 0;
        env->stage = CURRICULUM_TAIL_CHASE;  // Stage managed globally by Python
        env->curriculum_target = 0.0f;       // Start at stage 0
        if (DEBUG >= 1) {
            fprintf(stderr, "[INIT] FIRST init ptr=%p env_num=%d - setting total_episodes=0, stage=0\n", (void*)env, env_num);
        }
    } else {
        if (DEBUG >= 1) {
            fprintf(stderr, "[INIT] RE-init ptr=%p env_num=%d - preserving total_episodes=%d, stage=%d\n",
                    (void*)env, env_num, env->total_episodes, env->stage);
        }
    }
    env->is_initialized = 1;
    env->total_aileron_usage = 0.0f;

    // Initialize previous actions for control rate penalty
    env->prev_elevator = 0.0f;
    env->prev_aileron = 0.0f;
    env->prev_rudder = 0.0f;

    // Initialize flight physics parameters to defaults
    env->flight_params = default_flight_params();

    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));

    // Self-play: default to autopilot-controlled opponent
    env->use_opponent_override = 0;
    memset(env->opponent_actions_override, 0, sizeof(env->opponent_actions_override));

    // Opponent buffers: NULL by default, set by Python if dual self-play is enabled
    env->opponent_observations = NULL;
    env->opponent_rewards = NULL;

    // Eval spawn mode: 0 = random (default)
    env->eval_spawn_mode = 0;

    // Late-training debug logging: disabled by default
    env->global_step = 0;
    env->debug_trigger_step = 0;
    env->debug_log_file = NULL;
    env->debug_log_initialized = 0;

    // Opponent recovery hijacking: disabled by default, enabled by Python in self-play
    env->selfplay_active = 0;
    env->opponent_recovery_active = 0;
    env->opponent_recovery_tick_start = 0;
    // Config values set by binding.c from INI file
    env->recovery_altitude_threshold = 750.0f;  // Higher threshold for high-speed dives (was 500)
    env->recovery_trigger_prob = 0.5f;  // 50% chance to trigger recovery (was 10%)
    env->recovery_speed_threshold = 70.0f;
    env->recovery_bank_deg = 60.0f;
    env->recovery_rng_state = (unsigned int)rand();
    env->opponent_above_recovery_threshold = 1;  // Start assuming above threshold

    // Guided climb hijack: disabled by default
    env->guided_climb_active = 0;
    env->guided_climb_ticks_remaining = 0;
    env->guided_climb_elevator = 0.5f;  // Default: moderate pull for ~3G
}

void set_obs_highlight(Dogfight *env, int *indices, int count) {
    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));
    for (int i = 0; i < count && i < 25; i++) {
        if (indices[i] >= 0 && indices[i] < 25) {
            env->obs_highlight[indices[i]] = 1;
        }
    }
}

// Helper: set opponent reward (only if buffer exists, for dual self-play)
static inline void set_opponent_reward(Dogfight *env, float reward) {
    if (env->opponent_rewards != NULL) {
        env->opponent_rewards[0] = reward;
    }
}

// ============================================================================
// Late-training debug logging (activated when global_step >= debug_trigger_step)
// Logs comprehensive state data to /tmp/dogfight_debug_*.log for post-training analysis
// ============================================================================

static void init_debug_log(Dogfight* env) {
    if (env->debug_log_initialized) return;
    char filename[256];
    snprintf(filename, sizeof(filename), "/tmp/dogfight_debug_%d.log", env->env_num);
    env->debug_log_file = fopen(filename, "w");
    if (env->debug_log_file) {
        env->debug_log_initialized = 1;
        fprintf(env->debug_log_file, "# Dogfight Debug Log - env %d\n", env->env_num);
        fprintf(env->debug_log_file, "# Triggered at global_step >= %ld\n", env->debug_trigger_step);
        fprintf(env->debug_log_file, "# Format: per-tick state data, then episode summary\n\n");
        fflush(env->debug_log_file);
        fprintf(stderr, "[DEBUG-LOG] Opened %s for env %d\n", filename, env->env_num);
    } else {
        fprintf(stderr, "[DEBUG-LOG] ERROR: Failed to open %s\n", filename);
    }
}

static void debug_log_tick(Dogfight* env) {
    if (!env->debug_log_file) return;
    FILE* f = env->debug_log_file;
    Plane* p = &env->player;
    Plane* o = &env->opponent;

    // Only log every 10 ticks to reduce volume
    if (env->tick % 10 != 0) return;

    // Header for this tick
    fprintf(f, "\n=== STEP %ld TICK %d ===\n", env->global_step, env->tick);

    // Player state
    fprintf(f, "P_pos: %.1f,%.1f,%.1f\n", p->pos.x, p->pos.y, p->pos.z);
    fprintf(f, "P_vel: %.1f,%.1f,%.1f (speed=%.1f)\n",
            p->vel.x, p->vel.y, p->vel.z, norm3(p->vel));
    fprintf(f, "P_energy: %.1f\n", calc_specific_energy(p));
    fprintf(f, "P_g: %.2f\n", p->g_force);

    // Opponent state
    fprintf(f, "O_pos: %.1f,%.1f,%.1f\n", o->pos.x, o->pos.y, o->pos.z);
    fprintf(f, "O_vel: %.1f,%.1f,%.1f (speed=%.1f)\n",
            o->vel.x, o->vel.y, o->vel.z, norm3(o->vel));
    fprintf(f, "O_energy: %.1f\n", calc_specific_energy(o));

    // Relative geometry
    Vec3 rel_pos = sub3(o->pos, p->pos);
    float dist = norm3(rel_pos);
    Vec3 player_fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float aim_dot = dot3(normalize3(rel_pos), player_fwd);
    float aim_deg = acosf(clampf(aim_dot, -1.0f, 1.0f)) * RAD_TO_DEG;

    fprintf(f, "dist: %.1f\n", dist);
    fprintf(f, "aim_angle: %.1f deg\n", aim_deg);
    fprintf(f, "alt_diff: %.1f (P-O)\n", p->pos.z - o->pos.z);
    fprintf(f, "energy_diff: %.1f\n", calc_specific_energy(p) - calc_specific_energy(o));

    // Actions
    fprintf(f, "P_act: thr=%.2f elev=%.2f ail=%.2f rud=%.2f trig=%.2f\n",
            env->actions[0], env->actions[1], env->actions[2],
            env->actions[3], env->actions[4]);

    // Reward breakdown (all 10 terms)
    fprintf(f, "REWARDS:\n");
    fprintf(f, "  closing=%.5f aim=%.5f neg_g=%.5f stall=%.5f\n",
            env->r_closing, env->r_aim, env->r_neg_g, env->r_stall);
    fprintf(f, "  rudder=%.5f rate=%.5f altitude=%.5f time=%.6f\n",
            env->r_rudder, env->r_rate, env->r_altitude, env->r_time);
    fprintf(f, "  energy=%.5f energy_adv=%.5f\n",
            env->r_player_energy, env->r_energy_adv);
    fprintf(f, "  TOTAL=%.4f\n", env->rewards[0]);

    // Death spiral warning: both planes descending rapidly
    if (p->vel.z < -10.0f && o->vel.z < -10.0f) {
        fprintf(f, "SPIRAL: both descending P_vz=%.1f O_vz=%.1f\n", p->vel.z, o->vel.z);
    }

    fflush(f);
}

static void debug_log_episode_end(Dogfight* env) {
    if (!env->debug_log_file) return;
    FILE* f = env->debug_log_file;
    const char* death_names[] = {"NONE", "KILL", "OOB", "TIMEOUT", "SUPERSONIC"};

    fprintf(f, "\n=== EPISODE END at STEP %ld ===\n", env->global_step);
    fprintf(f, "ticks: %d\n", env->tick);
    fprintf(f, "episode_return: %.3f\n", env->episode_return);
    fprintf(f, "death_reason: %s (%d)\n", death_names[env->death_reason], env->death_reason);
    fprintf(f, "kill: %d, opp_kill: %d\n", env->kill, env->opp_kill);
    fprintf(f, "stage: %d\n", env->stage);

    // Flight envelope
    fprintf(f, "min_alt: %.1f, max_alt: %.1f\n", env->min_alt, env->max_alt);
    fprintf(f, "min_speed: %.1f, max_speed: %.1f\n", env->min_speed, env->max_speed);

    // Reward accumulators (all 10 terms)
    fprintf(f, "sum_r_closing: %.3f\n", env->sum_r_closing);
    fprintf(f, "sum_r_aim: %.3f\n", env->sum_r_aim);
    fprintf(f, "sum_r_neg_g: %.3f\n", env->sum_r_neg_g);
    fprintf(f, "sum_r_stall: %.3f\n", env->sum_r_speed);
    fprintf(f, "sum_r_rudder: %.3f\n", env->sum_r_rudder);
    fprintf(f, "sum_r_rate: %.3f\n", env->sum_r_rate);
    fprintf(f, "sum_r_altitude: %.3f\n", env->sum_r_altitude);
    fprintf(f, "sum_r_time: %.3f\n", env->sum_r_time);
    fprintf(f, "sum_r_player_energy: %.3f\n", env->sum_r_player_energy);
    fprintf(f, "sum_r_energy_adv: %.3f\n", env->sum_r_energy_adv);

    // Control analysis
    fprintf(f, "aileron_bias: %.1f\n", env->aileron_bias);
    float mean_control_rate = env->episode_control_rate / fmaxf((float)env->tick, 1.0f);
    fprintf(f, "mean_control_rate: %.4f\n", mean_control_rate);

    // Combat stats
    fprintf(f, "ticks_in_cone: %d\n", env->ticks_in_cone);
    fprintf(f, "closest_dist: %.1f\n", env->closest_dist);
    fprintf(f, "shots_fired: %.0f\n", env->episode_shots_fired);

    // Final positions
    fprintf(f, "final_P_pos: %.1f,%.1f,%.1f\n", env->player.pos.x, env->player.pos.y, env->player.pos.z);
    fprintf(f, "final_O_pos: %.1f,%.1f,%.1f\n", env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z);

    fprintf(f, "---\n");
    fflush(f);
}

void add_log(Dogfight *env) {
    // Level 1: Episode summary (one line, easy to grep)
    if (DEBUG >= 1 && env->env_num == 0) {
        const char* death_names[] = {"NONE", "KILL", "OOB", "TIMEOUT", "SUPERSONIC"};
        float mean_ail = env->total_aileron_usage / fmaxf((float)env->tick, 1.0f);
        printf("EP tick=%d ret=%.2f death=%s kill=%d stage=%d total_eps=%d mean_ail=%.2f bias=%.1f\n",
               env->tick, env->episode_return, death_names[env->death_reason],
               env->kill, env->stage, env->total_episodes, mean_ail, env->aileron_bias);
    }

    // Level 2: Reward breakdown (which components dominated?)
    if (DEBUG >= 2 && env->env_num == 0) {
        printf("  SHAPING: closing=%+.2f aim=%+.2f\n", env->sum_r_closing, env->sum_r_aim);
        printf("  PENALTY: stall=%.2f neg_g=%.2f rudder=%.2f rate=%.2f\n",
               env->sum_r_speed, env->sum_r_neg_g, env->sum_r_rudder, env->sum_r_rate);
        printf("  AIM: best=%.1f° in_cone=%d/%d (%.0f%%) closest=%.0fm\n",
               env->best_aim_angle * RAD_TO_DEG,
               env->ticks_in_cone, env->tick,
               100.0f * env->ticks_in_cone / fmaxf((float)env->tick, 1.0f),
               env->closest_dist);
    }

    // Level 3: Flight envelope and control statistics
    if (DEBUG >= 3 && env->env_num == 0) {
        float mean_throttle = env->sum_throttle / fmaxf((float)env->tick, 1.0f);
        printf("  FLIGHT: G=[%+.1f,%+.1f] bank=%.0f° pitch=%.0f° speed=[%.0f,%.0f] alt=[%.0f,%.0f]\n",
               env->min_g, env->max_g,
               env->max_bank * RAD_TO_DEG, env->max_pitch * RAD_TO_DEG,
               env->min_speed, env->max_speed,
               env->min_alt, env->max_alt);
        printf("  CONTROL: mean_throttle=%.0f%% trigger_pulls=%d shots=%d\n",
               mean_throttle * 100.0f, env->trigger_pulls, (int)env->episode_shots_fired);
    }

    if (DEBUG >= 10) printf("=== ADD_LOG ===\n");
    if (DEBUG >= 10) printf("  kill=%d, episode_return=%.2f, tick=%d\n", env->kill, env->episode_return, env->tick);
    if (DEBUG >= 10) printf("  episode_shots_fired=%.0f, reward=%.2f\n", env->episode_shots_fired, env->rewards[0]);
    env->log.episode_return += env->episode_return;
    env->log.episode_length += (float)env->tick;
    env->log.perf += env->kill ? 1.0f : 0.0f;
    // Self-play kill tracking: only when selfplay_active (set by Python at transition)
    if (env->selfplay_active) {
        env->log.sp_player_kills += env->kill ? 1.0f : 0.0f;
        env->log.sp_opp_kills += env->opp_kill ? 1.0f : 0.0f;
    }
    env->log.score += env->rewards[0];
    env->log.shots_fired += env->episode_shots_fired;
    env->log.accuracy = (env->log.shots_fired > 0.0f) ? (env->log.perf / env->log.shots_fired * 100.0f) : 0.0f;
    env->log.stage = (float)env->stage;

    env->log.total_stage_weight += STAGES[env->stage].weight; // coeffs to scale metrics based on difficulty
    env->log.total_abs_bias += fabsf(env->aileron_bias);
    env->log.stage_sum += (float)env->stage;  // Accumulate for avg_stage
    // Mean squared control delta per step this episode (lower = smoother control)
    env->log.total_control_rate += env->episode_control_rate / fmaxf((float)env->tick, 1.0f);

    // Track performance at MAJORITY stage (the one we're trying to master)
    // At target 0.9, majority is stage 1 (90% of episodes), not stage 0
    int mastery_stage = (int)(env->curriculum_target + 0.5f);  // round, not floor
    if (env->stage == mastery_stage) {
        env->log.base_stage_kills += env->kill ? 1.0f : 0.0f;
        env->log.base_stage_eps += 1.0f;
    }

    // Track clean fights (kills or timeouts, not ground crashes)
    // Clean fight = episode ended without either plane crashing into ground
    // NOTE: Only track during self-play (selfplay_active=1) to prevent fake ultimate2
    // During curriculum vs AutoAce, crashes are rare so clean_fights would be artificially high
    int is_clean = (env->death_reason == DEATH_KILL || env->death_reason == DEATH_TIMEOUT);
    if (env->selfplay_active) {
        env->log.clean_fights += is_clean ? 1.0f : 0.0f;
    }
    // During curriculum: clean_fights stays at 0, so ultimate2 = 0

    env->log.n += 1.0f;
    env->log.kill_rate = env->log.perf / fmaxf(env->log.n, 1.0f);
    env->log.avg_stage = env->log.stage_sum / env->log.n;
    env->log.avg_abs_bias = env->log.total_abs_bias / env->log.n;
    env->log.avg_stage_weight = env->log.total_stage_weight / env->log.n;

    // Ultimate = kill_rate * difficulty (no bias penalty)
    env->log.ultimate = env->log.kill_rate * env->log.avg_stage_weight;

    // Ultimate2 = kill_rate * clean_fight_rate (pure combat quality)
    // Penalizes death spirals by rewarding clean fights
    float clean_fight_rate = env->log.clean_fights / fmaxf(env->log.n, 1.0f);
    env->log.ultimate2 = env->log.kill_rate * clean_fight_rate;

    if (DEBUG >= 10) printf("  log.perf=%.2f, log.shots_fired=%.0f, log.n=%.0f\n", env->log.perf, env->log.shots_fired, env->log.n);
}

// ============================================================================
// Curriculum Learning: Stage-specific spawn functions
// ============================================================================

// Stage advancement handled in add_log() based on recent kill rate
CurriculumStage get_curriculum_stage(Dogfight *env) {
    if (!env->curriculum_enabled) return CURRICULUM_FULL_RANDOM;
    if (env->curriculum_randomize) {
        // Random stage for eval mode - tests all difficulties
        return (CurriculumStage)(rand() % CURRICULUM_COUNT);
    }

    // Probabilistic selection based on curriculum_target
    float target = env->curriculum_target;
    int base = (int)target;
    float frac = target - (float)base;

    if (base >= CURRICULUM_COUNT - 1) {
        return (CurriculumStage)(CURRICULUM_COUNT - 1);
    }

    // Probabilistic: if rand < frac, use base+1, else base
    if (rndf(0, 1) < frac) {
        return (CurriculumStage)(base + 1);
    }
    return (CurriculumStage)base;
}

// Stage 0: TAIL_CHASE - Opponent ahead, same heading (easiest)
static void spawn_tail_chase(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead with guaranteed minimum offset
    // At 300m, 5° gun cone = ~26m radius for hits
    // Minimum 26m y-offset guarantees ~5° at 300m (more at closer range)
    // Signed offset with minimum magnitude: either [-50, -26] or [26, 50]
    float y_sign = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float y_offset = y_sign * rndf(26, 50);

    // 20% chance: spawn player LOW (400m) with opponent ABOVE
    // Teaches altitude awareness early - don't descend with target
    if (rndf(0, 1) < 0.2f) {
        env->player.pos.z = 400.0f;  // Just below 500m recovery threshold
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(200, 400),
            player_pos.y + y_offset,
            700.0f + rndf(0, 200)  // Opponent 300-500m above player
        );
        reset_plane(&env->opponent, opp_pos, player_vel);
        env->opponent_ap.mode = AP_STRAIGHT;
        // More time for climb + pursuit in altitude-disadvantage variant
        env->max_steps = 2000;
        return;
    }

    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + y_offset,        // Min 26m = ~5° at 300m
        player_pos.z + rndf(-38, 38)    // z can still vary
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 1: HEAD_ON - Opponent coming toward us
static void spawn_head_on(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // 20% chance: spawn player LOW with opponent coming from ABOVE
    // Teaches: don't dive into head-on, maintain altitude
    if (rndf(0, 1) < 0.2f) {
        env->player.pos.z = 400.0f;  // Just below 500m recovery threshold
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(400, 600),
            player_pos.y + rndf(-50, 50),
            700.0f + rndf(0, 200)  // Opponent 300-500m above
        );
        Vec3 opp_vel = vec3(-player_vel.x, -player_vel.y, player_vel.z);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent_ap.mode = AP_STRAIGHT;
        // More time for climb + pursuit in altitude-disadvantage variant
        env->max_steps = 2000;
        return;
    }

    // Opponent 400-600m ahead, facing us (opposite velocity)
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(400, 600),
        player_pos.y + rndf(-50, 50),
        player_pos.z + rndf(-30, 30)
    );
    Vec3 opp_vel = vec3(-player_vel.x, -player_vel.y, player_vel.z);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 18: CROSSING - 45 degree deflection shots (reduced from 90° - see CURRICULUM_PLANS.md)
// 90° deflection is historically nearly impossible; 45° is achievable with proper lead
static void spawn_crossing(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 300-500m to the side, flying at 45° angle (not perpendicular)
    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(100, 200),
        player_pos.y + side * rndf(300, 500),
        player_pos.z + rndf(-50, 50)
    );
    // 45° crossing velocity: opponent flies at 45° angle across player's path
    // cos(45°) ≈ 0.707, sin(45°) ≈ 0.707
    float speed = norm3(player_vel);
    float cos45 = 0.7071f;
    float sin45 = 0.7071f;
    // side=+1 (right): fly toward (-45°) = (cos, -sin) to cross leftward
    // side=-1 (left): fly toward (+45°) = (cos, +sin) to cross rightward
    Vec3 opp_vel = vec3(speed * cos45, -side * speed * sin45, 0);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 2: VERTICAL - Above or below player
static void spawn_vertical(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead, 200-400m above OR below
    float vert = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float alt_offset = vert * rndf(200, 400);
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-50, 50),
        clampf(player_pos.z + alt_offset, 300, 4700)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_LEVEL;  // Maintain altitude

    // Speed boost only when opponent is ABOVE us (climbing needs energy, diving doesn't)
    if (opp_pos.z > player_pos.z) {
        env->player.vel = mul3(env->player.vel, 1.15f);
    }
}

// Stage 3: GENTLE_TURNS - Opponent does gentle turns (30°)
static void spawn_gentle_turns(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // 20% chance: spawn player LOW with opponent turning ABOVE
    // Teaches: climb while pursuing turning target
    if (rndf(0, 1) < 0.2f) {
        env->player.pos.z = 400.0f;  // Just below 500m recovery threshold
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(200, 500),
            player_pos.y + rndf(-100, 100),
            700.0f + rndf(0, 200)  // Opponent 300-500m above
        );
        reset_plane(&env->opponent, opp_pos, player_vel);
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
        // More time for climb + pursuit in altitude-disadvantage variant
        env->max_steps = 2000;
        return;
    }

    // Random spawn position (similar to original)
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 500),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    // Randomly choose turn direction - gentle 30° bank
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 4: OFFSET - Large lateral/vertical offset, same heading
// Teaches: Finding and tracking targets not directly in front
static void spawn_offset(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 150-300m ahead with LARGE lateral/vertical offset
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(150, 300),
        player_pos.y + rndf(-200, 200),   // Large lateral - can be way to the side
        clampf(player_pos.z + rndf(-150, 150), 300, 4700)  // Large vertical
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 5: ANGLED - Offset + different heading (±22°)
// Teaches: Pursuit geometry when target isn't flying your direction (small angle)
static void spawn_angled(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-150, 150),
        clampf(player_pos.z + rndf(-100, 100), 300, 4700)
    );

    // Heading offset: ±22° from player (reduced from ±45° for smoother progression)
    float heading_offset = rndf(-0.385f, 0.385f);  // ~22° in radians
    float player_heading = atan2f(player_vel.y, player_vel.x);
    float opp_heading = player_heading + heading_offset;

    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stages 6-9: Unified side spawn - uses angle_min_deg, angle_max_deg, bank from STAGES
// Stages 6-8: Target off axis, flying away (no turns)
// Stage 9: Same geometry + 30° turns
static void spawn_side(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    // 20% chance: ENERGY BUILDING scenario
    // Opponent VERY HIGH and SLOW - player MUST build energy over time to reach them
    // Can't just zoom climb - need sustained full throttle climbing for many seconds
    // Teaches: long-term energy planning, not just immediate pursuit
    if (rndf(0, 1) < 0.2f) {
        // Player at normal altitude, opponent 800-1200m ABOVE
        // This is too high to zoom climb - requires sustained energy building
        float opp_alt = player_pos.z + rndf(800, 1200);
        opp_alt = clampf(opp_alt, 1500, 4500);  // Keep in bounds

        // Opponent ahead and above, flying gentle turns at LOW throttle
        float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
        Vec3 opp_pos = vec3(
            player_pos.x + rndf(400, 700),
            player_pos.y + side * rndf(50, 200),
            opp_alt
        );

        // Opponent flying VERY SLOW (50% of player speed) - easy target IF you can reach them
        float player_speed = norm3(player_vel);
        float opp_speed = player_speed * 0.5f;
        float opp_heading = atan2f(player_vel.y, player_vel.x) + side * rndf(0.1f, 0.3f);
        Vec3 opp_vel = vec3(opp_speed * cosf(opp_heading), opp_speed * sinf(opp_heading), 0);

        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

        // Opponent: very gentle turns (15° bank), LOW throttle, bleeding energy
        // They're a sitting duck - the challenge is GETTING UP THERE
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = 15.0f * DEG_TO_RAD;  // Gentle 15° bank - won't go OOB
        env->opponent.throttle = 0.25f;  // Very low throttle - bleeding energy fast

        return;
    }

    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    float phi = rndf(-0.2f, 0.2f);  // ±11° elevation

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    float away_heading = azimuth;
    float opp_heading = away_heading + rndf(-0.35f, 0.35f);  // ±20° variance

    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

    // AP mode based on bank field: 0 = straight, >0 = turning
    if (cfg->bank > 0) {
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)cfg->bank * DEG_TO_RAD;
    } else {
        env->opponent_ap.mode = AP_STRAIGHT;
    }

    // Stages 8-9: Boost player speed 15% for pursuit advantage (wide angle chase)
    if (env->stage >= CURRICULUM_SIDE_FAR) {
        env->player.vel = mul3(env->player.vel, 1.15f);
    }

    // Speed boost when opponent is above (climbing needs energy)
    if (opp_pos.z > player_pos.z) {
        env->player.vel = mul3(env->player.vel, 1.15f);
    }
}

// Stage 10: DIVE_ATTACK - Player starts 500m above, 75° nose-down for fast catch-up
// Same spawn geometry as spawn_rear (90-150° off axis), but player has massive altitude/energy advantage
static void spawn_dive_attack(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    // Same azimuth geometry as spawn_rear (90-150° off axis)
    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    // Opponent spawns 500m BELOW player (big altitude advantage)
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z - 500 + rndf(-50, 50), 300, 4700)
    );

    float opp_heading = azimuth + rndf(-0.35f, 0.35f);  // ±20° variance
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_STRAIGHT : AP_LEVEL;

    // Player starts 75° nose down (same heading, just pitched)
    // Pitch rotation is around body Y-axis (right wing)
    // Positive pitch around Y = nose down in this coordinate system
    float pitch = 75.0f * DEG_TO_RAD;
    Quat pitch_quat = quat_from_axis_angle(vec3(0, 1, 0), pitch);
    env->player.ori = pitch_quat;
    // Velocity matches pitch direction (diving toward target area)
    env->player.vel = quat_rotate(pitch_quat, player_vel);
    env->player.prev_vel = env->player.vel;
}

// Stage 11: ZOOM_ATTACK - Player starts 500m below, 75° nose-up, near max speed
// Opposite of dive_attack: player zooms up toward target with high energy
static void spawn_zoom_attack(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    // Same azimuth geometry as spawn_rear (90-150° off axis)
    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    // Opponent spawns 300 ABOVE player (player zooms up)
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z + 300 + rndf(-50, 50), 300, 4700)
    );

    float opp_heading = azimuth + rndf(-0.35f, 0.35f);  // ±20° variance
    float opp_speed = norm3(player_vel);
    Vec3 opp_vel = vec3(opp_speed * cosf(opp_heading), opp_speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_STRAIGHT : AP_LEVEL;

    // Player starts 75° nose UP with near-max speed (~145 m/s)
    // Pitch rotation is around body Y-axis (right wing)
    // Negative pitch around Y = nose up in this coordinate system
    float pitch = -75.0f * DEG_TO_RAD;
    Quat pitch_quat = quat_from_axis_angle(vec3(0, 1, 0), pitch);
    env->player.ori = pitch_quat;

    // Set player to high speed (reduced from 140-150 due to instability at extreme pitch)
    float zoom_speed = rndf(110, 120);
    Vec3 base_vel = vec3(zoom_speed, 0, 0);
    env->player.vel = quat_rotate(pitch_quat, base_vel);
    env->player.prev_vel = env->player.vel;

    // ZOOM_ATTACK always has altitude disadvantage - needs more time for climb + pursuit
    env->max_steps = 4500;
}

// Stages 12-13: Unified rear spawn - uses angle_min_deg, angle_max_deg, bank from STAGES
// Stage 12: Target 90-150° off axis (rear quarters), 50/50 straight/level
// Stage 13: Same geometry + 30° turns (unchanged, zoom_attack inserted before these)
static void spawn_rear(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    const StageConfig* cfg = &STAGES[env->stage];

    float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float az_min = cfg->angle_min_deg * DEG_TO_RAD;
    float az_max = cfg->angle_max_deg * DEG_TO_RAD;
    float azimuth = side * rndf(az_min, az_max);

    float dist = rndf(300, 500);
    // Opponent spawns ~500m below player (large altitude advantage for rear chase)
    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth),
        player_pos.y + dist * sinf(azimuth),
        clampf(player_pos.z - 500 + rndf(-50, 50), 300, 4700)
    );

    float opp_heading = azimuth + rndf(-0.35f, 0.35f);  // ±20° variance
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

    // AP mode based on bank field: 0 = 50/50 straight/level, >0 = turning
    if (cfg->bank > 0) {
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)cfg->bank * DEG_TO_RAD;
    } else {
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_STRAIGHT : AP_LEVEL;
    }

    // Speed boost for rear chase - player starts faster to close the gap
    env->player.vel = mul3(env->player.vel, 1.25f);
    env->player.prev_vel = env->player.vel;
}

// Stage 14: FULL_PREDICTABLE - 360° spawn, heading correlated (flying away)
// Teaches: Full sphere awareness with predictable heading
static void spawn_full_predictable(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Full 360° spawn
    float azimuth = rndf(-M_PI, M_PI);
    float dist = rndf(300, 600);
    float phi = rndf(-0.3f, 0.3f);  // ±17° elevation

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(azimuth) * cosf(phi),
        player_pos.y + dist * sinf(azimuth) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    // KEY: Heading is CORRELATED - flying away from player
    float away_heading = azimuth;  // Same direction as spawn angle = flying away
    float opp_heading = away_heading + rndf(-0.52f, 0.52f);  // ±30° variance

    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 15: FULL_RANDOM - 360° spawn, random heading, 30° turns
// Teaches: Random heading (key difficulty!) - must read observation to determine velocity
static void spawn_full_random(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Random direction in 3D sphere (300-600m from player)
    float dist = rndf(300, 600);
    float theta = rndf(0, 2.0f * M_PI);      // Azimuth: 0-360°
    float phi = rndf(-0.3f, 0.3f);           // Elevation: ±17° (keep near level)

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    // Random velocity direction (not necessarily toward/away from player)
    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);

    // Set orientation to match velocity direction (yaw rotation around Z)
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // 3 modes: straight, level, turns (still 30° - steeper turns come in stage 16)
    float r = rndf(0, 1);
    if (r < 0.2f) env->opponent_ap.mode = AP_STRAIGHT;
    else if (r < 0.4f) env->opponent_ap.mode = AP_LEVEL;
    else env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;

    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 16: MEDIUM_TURNS - 360° spawn, random heading, 45° turns
// Teaches: Steeper 45° turns (first introduction of harder turns)
static void spawn_medium_turns(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Same geometry as FULL_RANDOM
    float dist = rndf(300, 600);
    float theta = rndf(0, 2.0f * M_PI);      // Azimuth: 0-360°
    float phi = rndf(-0.3f, 0.3f);           // Elevation: ±17° (keep near level)

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 4700)
    );

    // Random velocity direction (uncorrelated with position)
    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // 5 modes with 45° turns
    float r = rndf(0, 1);
    if (r < 0.2f) env->opponent_ap.mode = AP_STRAIGHT;
    else if (r < 0.4f) env->opponent_ap.mode = AP_LEVEL;
    else if (r < 0.6f) env->opponent_ap.mode = AP_TURN_LEFT;
    else if (r < 0.8f) env->opponent_ap.mode = AP_TURN_RIGHT;
    else env->opponent_ap.mode = AP_CLIMB;

    env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
}

// Stage 17: HARD_MANEUVERING - Hard turns (60°) and weave patterns
static void spawn_hard_maneuvering(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);

    // Pick from hard maneuver modes
    float r = rndf(0, 1);
    if (r < 0.3f) {
        env->opponent_ap.mode = AP_HARD_TURN_LEFT;
    } else if (r < 0.6f) {
        env->opponent_ap.mode = AP_HARD_TURN_RIGHT;
    } else {
        env->opponent_ap.mode = AP_WEAVE;
        env->opponent_ap.phase = rndf(0, 2.0f * M_PI);  // Random start phase
    }
}

// Stage 19: EVASIVE - Opponent reacts to player position (hardest)
static void spawn_evasive(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Override player altitude to near max (3500-4500m) for high-altitude combat
    env->player.pos.z = rndf(3500, 4500);
    player_pos.z = env->player.pos.z;  // Update local copy for opponent spawn

    // Spawn in various positions (like FULL_RANDOM)
    float dist = rndf(300, 500);
    float theta = rndf(0, 2.0f * M_PI);
    float phi = rndf(-0.3f, 0.3f);

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 2500, 4800)
    );

    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // Mix of hard modes with AP_EVASIVE dominant
    float r = rndf(0, 1);
    if (r < 0.4f) {
        env->opponent_ap.mode = AP_EVASIVE;
    } else if (r < 0.55f) {
        env->opponent_ap.mode = AP_HARD_TURN_LEFT;
    } else if (r < 0.7f) {
        env->opponent_ap.mode = AP_HARD_TURN_RIGHT;
    } else if (r < 0.85f) {
        env->opponent_ap.mode = AP_WEAVE;
        env->opponent_ap.phase = rndf(0, 2.0f * M_PI);
    } else {
        // 15% chance of regular turn modes (still steep 60°)
        env->opponent_ap.mode = rndf(0,1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = (float)STAGES[env->stage].bank * DEG_TO_RAD;
    }
}

// Stage 20: AUTOACE - Intelligent adversarial opponent (two-way combat)
static void spawn_autoace(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Override player altitude to mid-high (2500-4000m)
    env->player.pos.z = rndf(2500, 4000);
    player_pos.z = env->player.pos.z;

    // Spawn opponent in various positions (360 degree, varied distance)
    float dist = rndf(400, 700);
    float theta = rndf(0, 2.0f * M_PI);
    float phi = rndf(-0.25f, 0.25f);

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 2000, 4500)
    );

    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    env->opponent_ap.mode = AP_PURSUIT_LAG;
    autoace_init(&env->opponent_ace);
}

// EVAL spawn: True randomization with alternating advantages
// Used when curriculum_randomize=1 - creates varied, fair combat scenarios
static void spawn_eval_random(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Always clear head-on lockout first (only set if we choose head-on spawn)
    env->head_on_lockout = 0;
    env->prev_rel_dot = 0.0f;

    // Alternate who gets advantage based on episode count
    int player_advantage = (env->total_episodes % 2 == 0);

    // Random spawn type distribution:
    // 40% - tactical (one behind/side of other)
    // 30% - neutral (both at angles, neither clearly advantaged)
    // 20% - energy (altitude/speed difference)
    // 10% - head-on (with gun lockout until pass)
    float spawn_roll = rndf(0, 1);

    // Base altitude for combat (mid-altitude)
    float base_alt = rndf(2000, 3500);
    env->player.pos.z = base_alt;
    player_pos.z = base_alt;
    float speed = norm3(player_vel);

    if (spawn_roll < 0.40f) {
        // TACTICAL: One plane behind/side of other (clear advantage)
        float dist = rndf(300, 600);
        float angle_off = rndf(120, 180) * DEG_TO_RAD;  // Behind (120-180° off nose)
        float side = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;

        if (player_advantage) {
            // Player behind opponent - player has advantage
            float opp_heading = rndf(0, 2.0f * M_PI);
            Vec3 opp_pos = vec3(
                player_pos.x + rndf(300, 500),
                player_pos.y + side * rndf(50, 150),
                clampf(player_pos.z + rndf(-100, 100), 500, 4500)
            );
            Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
            reset_plane(&env->opponent, opp_pos, opp_vel);
            env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
            // Player heading toward opponent
            Vec3 to_opp = sub3(opp_pos, player_pos);
            float player_heading = atan2f(to_opp.y, to_opp.x);
            env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
            env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        } else {
            // Opponent behind player - opponent has advantage
            float player_heading = rndf(0, 2.0f * M_PI);
            env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
            env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
            // Opponent behind
            Vec3 opp_pos = vec3(
                player_pos.x - cosf(player_heading) * dist + side * sinf(player_heading) * rndf(50, 150),
                player_pos.y - sinf(player_heading) * dist - side * cosf(player_heading) * rndf(50, 150),
                clampf(player_pos.z + rndf(-100, 100), 500, 4500)
            );
            Vec3 to_player = sub3(player_pos, opp_pos);
            float opp_heading = atan2f(to_player.y, to_player.x);
            Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
            reset_plane(&env->opponent, opp_pos, opp_vel);
            env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
        }
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = rndf(30, 60) * DEG_TO_RAD;

    } else if (spawn_roll < 0.70f) {
        // NEUTRAL: Both at angles, converging - fair fight
        float dist = rndf(400, 700);
        float theta = rndf(0, 2.0f * M_PI);
        Vec3 opp_pos = vec3(
            player_pos.x + dist * cosf(theta),
            player_pos.y + dist * sinf(theta),
            clampf(player_pos.z + rndf(-200, 200), 500, 4500)
        );
        // Both heading toward a point between them (converging)
        Vec3 midpoint = mul3(add3(player_pos, opp_pos), 0.5f);
        Vec3 player_to_mid = sub3(midpoint, player_pos);
        Vec3 opp_to_mid = sub3(midpoint, opp_pos);
        // Add some angle offset so they're not perfectly converging
        float player_heading = atan2f(player_to_mid.y, player_to_mid.x) + rndf(-0.5f, 0.5f);
        float opp_heading = atan2f(opp_to_mid.y, opp_to_mid.x) + rndf(-0.5f, 0.5f);

        env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
        env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
        env->opponent_ap.target_bank = rndf(30, 45) * DEG_TO_RAD;

    } else if (spawn_roll < 0.90f) {
        // ENERGY: Altitude or speed advantage
        float dist = rndf(400, 600);
        float theta = rndf(0, 2.0f * M_PI);
        float alt_diff = rndf(300, 600);  // Significant altitude difference

        Vec3 opp_pos;
        if (player_advantage) {
            // Player higher (energy advantage)
            env->player.pos.z = base_alt + alt_diff;
            player_pos.z = env->player.pos.z;
            opp_pos = vec3(
                player_pos.x + dist * cosf(theta),
                player_pos.y + dist * sinf(theta),
                base_alt
            );
        } else {
            // Opponent higher (energy advantage)
            opp_pos = vec3(
                player_pos.x + dist * cosf(theta),
                player_pos.y + dist * sinf(theta),
                base_alt + alt_diff
            );
        }
        // Random headings
        float player_heading = rndf(0, 2.0f * M_PI);
        float opp_heading = rndf(0, 2.0f * M_PI);
        env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
        env->opponent_ap.mode = AP_PURSUIT_LEAD;  // Aggressive pursuit for energy fights

    } else {
        // HEAD-ON: Facing each other (rare, 10%) - guns locked until they pass
        float dist = rndf(600, 900);  // Start further apart
        float theta = rndf(0, 2.0f * M_PI);

        Vec3 opp_pos = vec3(
            player_pos.x + dist * cosf(theta),
            player_pos.y + dist * sinf(theta),
            clampf(player_pos.z + rndf(-100, 100), 500, 4500)
        );
        // Player faces opponent
        Vec3 to_opp = sub3(opp_pos, player_pos);
        float player_heading = atan2f(to_opp.y, to_opp.x);
        // Opponent faces player (opposite direction)
        float opp_heading = player_heading + M_PI;

        env->player.vel = vec3(speed * cosf(player_heading), speed * sinf(player_heading), 0);
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), player_heading);
        Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

        // HEAD-ON LOCKOUT: Disable guns until they pass each other
        env->head_on_lockout = 1;
        // Initialize tracking for pass detection
        Vec3 rel_pos = sub3(opp_pos, player_pos);
        Vec3 rel_vel = sub3(opp_vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos, rel_vel);

        env->opponent_ap.mode = AP_STRAIGHT;  // Fly straight initially
        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-SPAWN] Head-on spawn - guns locked until pass\n");
        }
    }

    // Reset autopilot PID state
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;

    if (DEBUG >= 1) {
        fprintf(stderr, "[EVAL-SPAWN] ep=%d advantage=%s spawn_type=%.0f%% dist=%.0fm\n",
                env->total_episodes, player_advantage ? "PLAYER" : "OPPONENT",
                spawn_roll * 100, norm3(sub3(env->opponent.pos, env->player.pos)));
    }
}

// Test spawn: Opponent behind player with advantage but not instant kill
// Player is 30° off opponent's nose - opponent must maneuver to get the shot
// Opponent is 400m behind, clear positional advantage
static void spawn_eval_opponent_advantage(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    env->head_on_lockout = 0;
    env->prev_rel_dot = 0.0f;

    // Player at base altitude, flying straight along +X
    float base_alt = 2500.0f;
    float speed = norm3(player_vel);
    if (speed < 70.0f) speed = 80.0f;

    env->player.pos = vec3(0, 0, base_alt);
    env->player.vel = vec3(speed, 0, 0);
    env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), 0.0f);  // Flying +X
    env->player.throttle = 0.5f;

    // Opponent 400m behind player
    float dist = 400.0f;
    Vec3 opp_pos = vec3(-dist, 0, base_alt);  // Directly behind player

    // Opponent heading: 30° off from pointing at player
    // Player is at (0,0), opponent at (-400,0)
    // Direct heading to player would be 0° (pointing +X)
    // We offset 30° so player is 30° off opponent's nose
    float angle_off_nose = 30.0f * DEG_TO_RAD;
    float opp_heading = angle_off_nose;  // Pointing 30° left of player

    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);
    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);
    env->opponent.throttle = 0.6f;

    // Autopilot: pursuit mode to track player
    env->opponent_ap.mode = AP_PURSUIT_LEAD;
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;

    if (DEBUG >= 1) {
        Vec3 to_player = sub3(env->player.pos, opp_pos);
        float actual_dist = norm3(to_player);
        Vec3 opp_fwd = quat_rotate(env->opponent.ori, vec3(1, 0, 0));
        Vec3 to_player_norm = normalize3(to_player);
        float aim_dot = dot3(opp_fwd, to_player_norm);
        float aim_angle = acosf(clampf(aim_dot, -1.0f, 1.0f)) * RAD_TO_DEG;
        fprintf(stderr, "[EVAL-OPP-ADV] dist=%.0fm aim_angle=%.1f° (cone=5°)\n",
                actual_dist, aim_angle);
    }
}

// EVAL spawn mode 2: Symmetric scenario pool for fair Elo evaluation
// Randomly selects from 3 scenarios: head-on merge, post-merge zoom, turning fight
// All scenarios are symmetric with slight perturbations to break identical observations
static void spawn_eval_merge(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float speed = norm3(player_vel);
    if (speed < 70.0f) speed = 80.0f;

    // Shared: random center altitude and merge axis
    float base_alt = rndf(2500, 3500);
    float theta = rndf(0, 2.0f * M_PI);  // merge axis heading

    // Tiny asymmetric perturbations (break identical obs, no real advantage)
    float pos_jitter = rndf(-5, 5);
    float alt_jitter = rndf(-5, 5);
    float speed_jitter = rndf(-3, 3);
    float angle_jitter = rndf(-0.035f, 0.035f);  // ~±2°

    int scenario = (int)(rndf(0, 2.999f));  // 0, 1, or 2

    if (scenario == 0) {
        // === Scenario 1: Head-On Merge ===
        // Classic merge. Both approaching, guns locked until pass.
        float half_dist = rndf(300, 450);
        float p_speed = speed + speed_jitter;
        float o_speed = speed - speed_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        float p_heading = theta + angle_jitter;
        float o_heading = theta + (float)M_PI - angle_jitter;

        // Player
        env->player.pos = p_pos;
        env->player.ori = quat_from_axis_angle(vec3(0, 0, 1), p_heading);
        env->player.vel = vec3(p_speed * cosf(p_heading), p_speed * sinf(p_heading), 0);

        // Opponent
        Vec3 opp_vel = vec3(o_speed * cosf(o_heading), o_speed * sinf(o_heading), 0);
        reset_plane(&env->opponent, opp_pos, opp_vel);
        env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), o_heading);

        env->head_on_lockout = 1;

        // Initialize pass detection tracking
        Vec3 rel_pos = sub3(opp_pos, p_pos);
        Vec3 rel_vel = sub3(opp_vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos, rel_vel);

        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-MERGE] scenario=HEAD_ON dist=%.0fm alt=%.0fm heading=%.1f°\n",
                    half_dist * 2, base_alt, theta * RAD_TO_DEG);
        }

    } else if (scenario == 1) {
        // === Scenario 2: Post-Merge Zoom ===
        // Both just passed and pulled up. Who manages energy better?
        // Flying AWAY from each other, both climbing nose-up.
        float half_dist = rndf(100, 200);
        float pitch_angle = rndf(30, 50) * DEG_TO_RAD;
        float zoom_speed = rndf(70, 90);
        float p_speed = zoom_speed + speed_jitter;
        float o_speed = zoom_speed - speed_jitter;

        // Positions: separated, backs to each other
        // Player flies along +theta, opponent flies along +theta+PI (away from each other)
        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Player heading: away from opponent (along +theta direction)
        float p_heading = theta + angle_jitter;
        // Opponent heading: away from player (along +theta+PI direction)
        float o_heading = theta + (float)M_PI - angle_jitter;

        // Orientation: heading rotation, then pitch up
        // Compose: pitch around body Y, then heading around world Z
        Quat p_heading_q = quat_from_axis_angle(vec3(0, 0, 1), p_heading);
        Quat p_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);  // negative = nose up (Z up convention)
        Quat p_ori = quat_mul(p_heading_q, p_pitch_q);

        Quat o_heading_q = quat_from_axis_angle(vec3(0, 0, 1), o_heading);
        Quat o_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);
        Quat o_ori = quat_mul(o_heading_q, o_pitch_q);

        // Velocity aligned with nose direction
        Vec3 p_vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);
        Vec3 o_vel = mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed);

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = p_vel;

        reset_plane(&env->opponent, opp_pos, o_vel);
        env->opponent.ori = o_ori;

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-MERGE] scenario=POST_MERGE_ZOOM dist=%.0fm alt=%.0fm pitch=%.0f° heading=%.1f°\n",
                    half_dist * 2, base_alt, pitch_angle * RAD_TO_DEG, theta * RAD_TO_DEG);
        }

    } else {
        // === Scenario 3: Turning Fight ===
        // Engaged in a turning fight. Both banked, pulling toward each other.
        float half_dist = rndf(150, 250);
        float bank_angle = rndf(45, 60) * DEG_TO_RAD;
        float pitch_angle = rndf(5, 10) * DEG_TO_RAD;
        float turn_speed = rndf(70, 85);
        float p_speed = turn_speed + speed_jitter;
        float o_speed = turn_speed - speed_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Both heading roughly toward each other, but offset ~45° to simulate a turn
        float turn_offset = rndf(30, 60) * DEG_TO_RAD;
        float p_heading = theta + turn_offset + angle_jitter;
        float o_heading = theta + (float)M_PI - turn_offset - angle_jitter;

        // Player banks left (toward opponent), opponent banks right (toward player)
        // Since they face each other, mirrored bank = same direction of turn
        Quat p_heading_q = quat_from_axis_angle(vec3(0, 0, 1), p_heading);
        Quat p_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);
        Quat p_bank_q = quat_from_axis_angle(vec3(1, 0, 0), -bank_angle);  // bank left
        Quat p_ori = quat_mul(p_heading_q, quat_mul(p_pitch_q, p_bank_q));

        Quat o_heading_q = quat_from_axis_angle(vec3(0, 0, 1), o_heading);
        Quat o_pitch_q = quat_from_axis_angle(vec3(0, 1, 0), -pitch_angle);
        Quat o_bank_q = quat_from_axis_angle(vec3(1, 0, 0), bank_angle);   // bank right (mirrored)
        Quat o_ori = quat_mul(o_heading_q, quat_mul(o_pitch_q, o_bank_q));

        Vec3 p_vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);
        Vec3 o_vel = mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed);

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = p_vel;

        reset_plane(&env->opponent, opp_pos, o_vel);
        env->opponent.ori = o_ori;

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1) {
            fprintf(stderr, "[EVAL-MERGE] scenario=TURNING_FIGHT dist=%.0fm alt=%.0fm bank=%.0f° heading=%.1f°\n",
                    half_dist * 2, base_alt, bank_angle * RAD_TO_DEG, theta * RAD_TO_DEG);
        }
    }
}

static void spawn_eval_midfight(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    float speed = norm3(player_vel);
    if (speed < 70.0f) speed = 80.0f;

    float base_alt = rndf(2500, 3500);
    float theta = rndf(0, 2.0f * M_PI);  // merge axis heading

    // Tiny asymmetric perturbations
    float pos_jitter = rndf(-5, 5);
    float alt_jitter = rndf(-5, 5);
    float speed_jitter = rndf(-3, 3);
    float angle_jitter = rndf(-0.035f, 0.035f);  // ~±2°

    // Alternate who gets which role
    int swap_roles = (env->total_episodes % 2);

    int scenario = (int)(rndf(0, 4.999f));  // 0-4

    if (scenario == 0) {
        // === Rolling Scissors ===
        // Crossing paths, hard banks opposite directions, both pulling up
        float half_dist = rndf(75, 125);
        float bank = rndf(60, 80) * DEG_TO_RAD;
        float pitch = rndf(15, 25) * DEG_TO_RAD;
        float scr_speed = rndf(65, 75);
        float p_speed = scr_speed + speed_jitter;
        float o_speed = scr_speed - speed_jitter;

        // Crossing angle: ~60-90° off from head-on
        float cross_offset = rndf(30, 45) * DEG_TO_RAD;
        float p_heading = theta + cross_offset + angle_jitter;
        float o_heading = theta + (float)M_PI - cross_offset - angle_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Player banks left, opponent banks right (crossing)
        float p_bank = swap_roles ? bank : -bank;
        float o_bank = swap_roles ? -bank : bank;

        Quat p_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), p_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), p_bank)));
        Quat o_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), o_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), o_bank)));

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);

        reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed));
        env->opponent.ori = o_ori;

        env->head_on_lockout = 1;
        Vec3 rel_pos_sc0 = sub3(env->opponent.pos, env->player.pos);
        Vec3 rel_vel_sc0 = sub3(env->opponent.vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos_sc0, rel_vel_sc0);

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=ROLLING_SCISSORS dist=%.0fm alt=%.0fm lockout=1\n",
                    half_dist * 2, base_alt);

    } else if (scenario == 1) {
        // === High Yo-Yo ===
        // Attacker above pulling down, defender turning hard below
        float alt_sep = rndf(300, 500);
        float horiz_dist = rndf(200, 400);
        float atk_pitch = -rndf(25, 35) * DEG_TO_RAD;  // nose down
        float atk_bank = rndf(30, 50) * DEG_TO_RAD;
        float def_bank = rndf(50, 65) * DEG_TO_RAD;
        float atk_speed = rndf(85, 95);
        float def_speed = rndf(70, 80);

        Vec3 hi_pos = vec3(
            player_pos.x - horiz_dist * cosf(theta) + pos_jitter,
            player_pos.y - horiz_dist * sinf(theta),
            clampf(base_alt + alt_sep / 2 - alt_jitter, 500, 4500)
        );
        Vec3 lo_pos = vec3(
            player_pos.x + horiz_dist * cosf(theta) - pos_jitter,
            player_pos.y + horiz_dist * sinf(theta),
            clampf(base_alt - alt_sep / 2 + alt_jitter, 500, 4500)
        );

        // Attacker: nose down + banked, heading toward defender
        float atk_heading = theta + angle_jitter;
        Quat atk_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), atk_heading),
                       quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -atk_pitch),
                                quat_from_axis_angle(vec3(1, 0, 0), -atk_bank)));

        // Defender: level, hard bank turn (perpendicular to merge axis)
        float def_heading = theta + (float)M_PI / 2 + angle_jitter;
        Quat def_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), def_heading),
                                quat_from_axis_angle(vec3(1, 0, 0), -def_bank));

        Vec3 *p_pos_ptr, *o_pos_ptr;
        Quat p_ori, o_ori;
        float p_speed, o_speed;
        if (swap_roles) {
            p_pos_ptr = &lo_pos; o_pos_ptr = &hi_pos;
            p_ori = def_ori; o_ori = atk_ori;
            p_speed = def_speed + speed_jitter; o_speed = atk_speed - speed_jitter;
        } else {
            p_pos_ptr = &hi_pos; o_pos_ptr = &lo_pos;
            p_ori = atk_ori; o_ori = def_ori;
            p_speed = atk_speed + speed_jitter; o_speed = def_speed - speed_jitter;
        }

        env->player.pos = *p_pos_ptr;
        env->player.ori = p_ori;
        env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);

        reset_plane(&env->opponent, *o_pos_ptr, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed));
        env->opponent.ori = o_ori;

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=HIGH_YOYO alt_sep=%.0fm horiz=%.0fm\n",
                    alt_sep, horiz_dist);

    } else if (scenario == 2) {
        // === Overshoot ===
        // One just overshot, scrambling to re-engage. Other reversing behind.
        float along_dist = rndf(150, 250);  // how far ahead the overshooting plane is
        float behind_dist = rndf(100, 200);
        float overshoot_speed = rndf(95, 110);
        float reversal_speed = rndf(70, 80);
        float reversal_bank = rndf(55, 70) * DEG_TO_RAD;

        // Overshooting plane: flying straight past, wings level
        float fwd_heading = theta + angle_jitter;
        Vec3 fwd_pos = vec3(
            player_pos.x + along_dist * cosf(theta) + pos_jitter,
            player_pos.y + along_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Quat fwd_ori = quat_from_axis_angle(vec3(0, 0, 1), fwd_heading);
        Vec3 fwd_vel = vec3(overshoot_speed * cosf(fwd_heading), overshoot_speed * sinf(fwd_heading), 0);

        // Reversing plane: behind, in hard bank reversal turn
        float rev_heading = theta + rndf(0.35f, 0.70f);  // ~20-40° off from straight chase
        Vec3 rev_pos = vec3(
            player_pos.x - behind_dist * cosf(theta) - pos_jitter,
            player_pos.y - behind_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );
        Quat rev_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), rev_heading),
                                quat_from_axis_angle(vec3(1, 0, 0), -reversal_bank));
        Vec3 rev_vel = mul3(quat_rotate(rev_ori, vec3(1, 0, 0)), reversal_speed);

        if (swap_roles) {
            env->player.pos = fwd_pos;
            env->player.ori = fwd_ori;
            env->player.vel = fwd_vel;
            reset_plane(&env->opponent, rev_pos, rev_vel);
            env->opponent.ori = rev_ori;
        } else {
            env->player.pos = rev_pos;
            env->player.ori = rev_ori;
            env->player.vel = rev_vel;
            reset_plane(&env->opponent, fwd_pos, fwd_vel);
            env->opponent.ori = fwd_ori;
        }

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=OVERSHOOT fwd=%.0fm behind=%.0fm\n",
                    along_dist, behind_dist);

    } else if (scenario == 3) {
        // === Vertical Fight ===
        // Both climbing in a vertical rolling engagement
        float half_dist = rndf(100, 150);
        float pitch = rndf(50, 70) * DEG_TO_RAD;
        float bank = rndf(25, 35) * DEG_TO_RAD;
        float climb_speed = rndf(75, 85);
        float p_speed = climb_speed + speed_jitter;
        float o_speed = climb_speed - speed_jitter;

        Vec3 p_pos = vec3(
            player_pos.x - half_dist * cosf(theta) + pos_jitter,
            player_pos.y - half_dist * sinf(theta),
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Vec3 opp_pos = vec3(
            player_pos.x + half_dist * cosf(theta) - pos_jitter,
            player_pos.y + half_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );

        // Both climbing, banked opposite directions
        float p_heading = theta + rndf(-0.17f, 0.17f) + angle_jitter;  // ~±10° heading spread
        float o_heading = theta + (float)M_PI + rndf(-0.17f, 0.17f) - angle_jitter;

        float p_bank = swap_roles ? bank : -bank;
        float o_bank = swap_roles ? -bank : bank;

        Quat p_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), p_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), p_bank)));
        Quat o_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), o_heading),
                     quat_mul(quat_from_axis_angle(vec3(0, 1, 0), -pitch),
                              quat_from_axis_angle(vec3(1, 0, 0), o_bank)));

        env->player.pos = p_pos;
        env->player.ori = p_ori;
        env->player.vel = mul3(quat_rotate(p_ori, vec3(1, 0, 0)), p_speed);

        reset_plane(&env->opponent, opp_pos, mul3(quat_rotate(o_ori, vec3(1, 0, 0)), o_speed));
        env->opponent.ori = o_ori;

        env->head_on_lockout = 1;
        Vec3 rel_pos_sc3 = sub3(env->opponent.pos, env->player.pos);
        Vec3 rel_vel_sc3 = sub3(env->opponent.vel, env->player.vel);
        env->prev_rel_dot = dot3(rel_pos_sc3, rel_vel_sc3);

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=VERTICAL pitch=%.0f° bank=%.0f° dist=%.0fm lockout=1\n",
                    pitch * RAD_TO_DEG, bank * RAD_TO_DEG, half_dist * 2);

    } else {
        // === Split-S Entry ===
        // One inverted pulling through, other pursuing
        float sep_dist = rndf(300, 400);
        float inv_speed = rndf(80, 90);
        float pursue_speed = rndf(75, 85);
        float inv_pitch = rndf(5, 15) * DEG_TO_RAD;  // slightly nose-down
        float pursue_bank = rndf(25, 35) * DEG_TO_RAD;

        // Inverted plane: ahead, upside down, slightly nose-down
        float inv_heading = theta + angle_jitter;
        Vec3 inv_pos = vec3(
            player_pos.x + pos_jitter,
            player_pos.y,
            clampf(base_alt - alt_jitter, 500, 4500)
        );
        Quat inv_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), inv_heading),
                       quat_mul(quat_from_axis_angle(vec3(1, 0, 0), (float)M_PI),  // inverted
                                quat_from_axis_angle(vec3(0, 1, 0), inv_pitch)));   // nose-down when inverted

        // Pursuer: behind, banked, chasing
        float pursue_heading = theta - angle_jitter;
        Vec3 pursue_pos = vec3(
            player_pos.x - sep_dist * cosf(theta) - pos_jitter,
            player_pos.y - sep_dist * sinf(theta),
            clampf(base_alt + alt_jitter, 500, 4500)
        );
        Quat pursue_ori = quat_mul(quat_from_axis_angle(vec3(0, 0, 1), pursue_heading),
                                   quat_from_axis_angle(vec3(1, 0, 0), -pursue_bank));

        if (swap_roles) {
            env->player.pos = pursue_pos;
            env->player.ori = pursue_ori;
            env->player.vel = mul3(quat_rotate(pursue_ori, vec3(1, 0, 0)), pursue_speed + speed_jitter);
            reset_plane(&env->opponent, inv_pos, mul3(quat_rotate(inv_ori, vec3(1, 0, 0)), inv_speed - speed_jitter));
            env->opponent.ori = inv_ori;
        } else {
            env->player.pos = inv_pos;
            env->player.ori = inv_ori;
            env->player.vel = mul3(quat_rotate(inv_ori, vec3(1, 0, 0)), inv_speed + speed_jitter);
            reset_plane(&env->opponent, pursue_pos, mul3(quat_rotate(pursue_ori, vec3(1, 0, 0)), pursue_speed - speed_jitter));
            env->opponent.ori = pursue_ori;
        }

        env->head_on_lockout = 0;
        env->prev_rel_dot = 0.0f;

        if (DEBUG >= 1)
            fprintf(stderr, "[EVAL-MIDFIGHT] scenario=SPLIT_S sep=%.0fm alt=%.0fm\n",
                    sep_dist, base_alt);
    }
}

// Master spawn function: dispatches to stage-specific spawner
void spawn_by_curriculum(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // For eval mode (curriculum_randomize=1), use spawn based on eval_spawn_mode
    if (env->curriculum_randomize) {
        if (env->eval_spawn_mode == 1) {
            // Mode 1: opponent advantage - for testing if opponent can kill player
            spawn_eval_opponent_advantage(env, player_pos, player_vel);
        } else if (env->eval_spawn_mode == 2) {
            // Mode 2: symmetric merge - fair Elo evaluation
            spawn_eval_merge(env, player_pos, player_vel);
        } else if (env->eval_spawn_mode == 3) {
            // Mode 3: mid-fight scenarios - banked/pitched engaged orientations
            spawn_eval_midfight(env, player_pos, player_vel);
        } else {
            // Mode 0 (default): random spawn
            spawn_eval_random(env, player_pos, player_vel);
        }
        // Eval mode uses stage 20 (AutoAce) max_steps for fair combat duration
        env->max_steps = STAGES[CURRICULUM_AUTOACE].max_steps;  // 6000
        return;
    }

    CurriculumStage new_stage = get_curriculum_stage(env);

    // Log stage transitions
    if (new_stage != env->stage) {
        if (DEBUG >= 1) {
            fprintf(stderr, "[STAGE_CHANGE] ptr=%p env=%d eps=%d: stage %d -> %d\n",
                   (void*)env, env->env_num, env->total_episodes, env->stage, new_stage);
            fflush(stderr);
        }
        env->stage = new_stage;
    }

    // Use function pointer from STAGES table (replaces 18-case switch)
    if (env->stage < CURRICULUM_COUNT) {
        STAGES[env->stage].spawn(env, player_pos, player_vel);

        // Use per-stage max_steps for advanced stages (8+) where episode length matters
        // Earlier stages use global max_steps from Python config for fast iteration
        // The original "training regression" was from variable episode lengths during early training
        // By stage 8+, agents are stable enough to handle longer episodes
        if (env->stage >= CURRICULUM_SIDE_FAR) {  // Stage 8+
            env->max_steps = STAGES[env->stage].max_steps;
        }
        // else: keep env->max_steps from Python init (already set)
    } else {
        spawn_evasive(env, player_pos, player_vel);  // Fallback for invalid stage
    }

    // Reset autopilot PID state after spawning
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;
}

// Legacy spawn (for curriculum_enabled=0)
void spawn_legacy(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 500),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);

    // Handle autopilot: randomize if configured, reset PID state
    if (env->opponent_ap.randomize_on_reset) {
        autopilot_randomize(&env->opponent_ap);
    }
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;
}

// ============================================================================
// Global curriculum control (called from Python based on aggregate kill_rate)
// ============================================================================

// Set curriculum stage for a single environment (used by vec version)
void set_curriculum_stage(Dogfight *env, int stage) {
    if (stage >= 0 && stage < CURRICULUM_COUNT) {
        env->stage = (CurriculumStage)stage;
        env->curriculum_target = (float)stage;  // Sync target for probabilistic selection
    }
}

// Set curriculum target (float 0.0-15.0) for probabilistic stage assignment
void set_curriculum_target(Dogfight *env, float target) {
    env->curriculum_target = fminf(fmaxf(target, 0.0f), (float)(CURRICULUM_COUNT - 1));
}

// ============================================================================

void c_reset(Dogfight *env) {
    // Save last episode result for rendering before reset
    env->last_death_reason = env->death_reason;
    if (env->death_reason == DEATH_KILL && env->kill) {
        env->last_winner = 1;   // Player won (got the kill)
    } else if (env->death_reason == DEATH_KILL) {
        env->last_winner = -1;  // Opponent won (player was killed)
    } else {
        env->last_winner = 0;   // Draw/timeout/OOB
    }

    // Curriculum stage is now managed globally by Python based on aggregate kill_rate
    // (see set_curriculum_stage() called from training loop)

    env->total_episodes++;

    env->tick = 0;
    env->episode_return = 0.0f;

    // Clear episode tracking (safe to clear kill after curriculum used it)
    env->kill = 0;
    env->opp_kill = 0;
    env->episode_shots_fired = 0.0f;
    env->total_aileron_usage = 0.0f;
    env->aileron_bias = 0.0f;
    env->episode_control_rate = 0.0f;

    // Reset reward accumulators
    env->sum_r_closing = 0.0f;
    env->sum_r_speed = 0.0f;
    env->sum_r_neg_g = 0.0f;
    env->sum_r_rudder = 0.0f;
    env->sum_r_aim = 0.0f;
    env->sum_r_rate = 0.0f;
    env->sum_r_altitude = 0.0f;
    env->sum_r_time = 0.0f;
    env->sum_r_player_energy = 0.0f;
    env->sum_r_energy_adv = 0.0f;
    env->death_reason = DEATH_NONE;

    // Reset per-tick reward fields
    env->r_closing = 0.0f;
    env->r_aim = 0.0f;
    env->r_neg_g = 0.0f;
    env->r_stall = 0.0f;
    env->r_rudder = 0.0f;
    env->r_rate = 0.0f;
    env->r_altitude = 0.0f;
    env->r_time = 0.0f;
    env->r_player_energy = 0.0f;
    env->r_energy_adv = 0.0f;

    // Reset aiming diagnostics
    env->best_aim_angle = M_PI;      // Start at worst (180°)
    env->ticks_in_cone = 0;
    env->closest_dist = 10000.0f;    // Start at max

    // Reset flight envelope diagnostics
    env->max_g = 1.0f;               // Start at 1G (level flight)
    env->min_g = 1.0f;
    env->max_bank = 0.0f;
    env->max_pitch = 0.0f;
    env->min_speed = 10000.0f;       // Start at max
    env->max_speed = 0.0f;
    env->min_alt = 10000.0f;         // Start at max
    env->max_alt = 0.0f;
    env->sum_throttle = 0.0f;
    env->trigger_pulls = 0;
    env->prev_trigger = 0;

    // Head-on lockout (only set by spawn_eval_random for head-on spawns)
    env->head_on_lockout = 0;
    env->prev_rel_dot = 0.0f;

    // Reset opponent recovery state (death spiral prevention)
    env->opponent_recovery_active = 0;
    env->opponent_recovery_tick_start = 0;
    env->opponent_above_recovery_threshold = 1;  // Ready to detect next crossing

    // Reset previous actions for control rate penalty
    env->prev_elevator = 0.0f;
    env->prev_aileron = 0.0f;
    env->prev_rudder = 0.0f;

    // Gun cone for hit detection - stays fixed at 5°
    env->cos_gun_cone = cosf(env->gun_cone_angle);

    // Spawn player at random position with base velocity
    // Use most of the sky (800-4200m) but avoid very low altitudes
    Vec3 pos = vec3(rndf(-500, 500), rndf(-500, 500), rndf(800, 4200));
    Vec3 vel = vec3(80, 0, 0);  // Base speed, will be randomized below
    reset_plane(&env->player, pos, vel);

    // Spawn opponent based on curriculum stage (or legacy if disabled)
    if (env->curriculum_enabled) {
        spawn_by_curriculum(env, pos, vel);

        // Phase 1: Apply stage-dependent speed randomization to both planes
        SpawnRandomization r = get_spawn_randomization(env->stage);
        float target_speed = rndf(r.speed_min, r.speed_max);
        float speed_ratio = target_speed / 80.0f;  // Scale from base speed
        env->player.vel = mul3(env->player.vel, speed_ratio);
        env->player.prev_vel = env->player.vel;  // Keep in sync
        env->opponent.vel = mul3(env->opponent.vel, speed_ratio);
        env->opponent.prev_vel = env->opponent.vel;

        // Phase 2: Apply stage-dependent throttle randomization
        env->player.throttle = rndf(r.throttle_min, r.throttle_max);
        env->opponent_ap.throttle = rndf(r.throttle_min, r.throttle_max);  // Autopilot throttle
    } else {
        spawn_legacy(env, pos, vel);
    }

    if (DEBUG >= 10) printf("=== RESET ===\n");
    if (DEBUG >= 10) printf("kill=%d, episode_shots_fired=%.0f (now cleared)\n", env->kill, env->episode_shots_fired);
    if (DEBUG >= 10) printf("player_pos=(%.1f, %.1f, %.1f)\n", pos.x, pos.y, pos.z);
    if (DEBUG >= 10) printf("player_vel=(%.1f, %.1f, %.1f) speed=%.1f\n", vel.x, vel.y, vel.z, norm3(vel));
    if (DEBUG >= 10) printf("opponent_pos=(%.1f, %.1f, %.1f)\n", env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z);
    if (DEBUG >= 10) printf("initial_dist=%.1f m, stage=%d\n", norm3(sub3(env->opponent.pos, pos)), env->stage);

    compute_observations(env);
#if DEBUG >= 5
    print_observations(env);
#endif
}

// Check if shooter hits target (cone-based hit detection)
bool check_hit(Plane *shooter, Plane *target, float cos_gun_cone) {
    Vec3 to_target = sub3(target->pos, shooter->pos);
    float dist = norm3(to_target);
    if (dist > GUN_RANGE) return false;
    if (dist < 1.0f) return false;  // Too close (avoid division issues)

    Vec3 forward = quat_rotate(shooter->ori, vec3(1, 0, 0));
    Vec3 to_target_norm = normalize3(to_target);
    float cos_angle = dot3(to_target_norm, forward);
    return cos_angle > cos_gun_cone;
}
void c_step(Dogfight *env) {
    env->tick++;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    if (DEBUG >= 10) printf("\n========== TICK %d ==========\n", env->tick);
    if (DEBUG >= 10) printf("=== ACTIONS ===\n");
    if (DEBUG >= 10) printf("throttle_raw=%.3f -> throttle=%.3f\n", env->actions[0], (env->actions[0] + 1.0f) * 0.5f);
    if (DEBUG >= 10) printf("elevator=%.3f -> pitch_rate=%.3f rad/s\n", env->actions[1], env->actions[1] * MAX_PITCH_RATE);
    if (DEBUG >= 10) printf("ailerons=%.3f -> roll_rate=%.3f rad/s\n", env->actions[2], env->actions[2] * MAX_ROLL_RATE);
    if (DEBUG >= 10) printf("rudder=%.3f -> yaw_rate=%.3f rad/s\n", env->actions[3], -env->actions[3] * MAX_YAW_RATE);
    if (DEBUG >= 10) printf("trigger=%.3f (fires if >0.5)\n", env->actions[4]);

    // Player uses full physics with actions (with runtime-configurable params)
    step_plane_with_params(&env->player, env->actions, DT, &env->flight_params);

    // === Opponent Recovery Hijacking (breaks death spiral equilibrium) ===
    // Only active during self-play (selfplay_active=1, set by Python when transitioning)
    // When opponent is low and descending, there's a chance to hijack controls for recovery
    Plane* opp = &env->opponent;

    // Recovery RNG (separate from main RNG to avoid affecting other randomization)
    #define RECOVERY_RAND() ({ \
        env->recovery_rng_state = env->recovery_rng_state * 1103515245 + 12345; \
        (float)((env->recovery_rng_state >> 16) & 0x7FFF) / 32767.0f; \
    })

    if (env->opponent_recovery_active) {
        // Check if recovery is complete
        int ticks_in_recovery = env->tick - env->opponent_recovery_tick_start;
        int recovery_complete = 0;

        // Complete if: gained enough altitude, or took too long, or in turn phase for a while
        if (opp->pos.z > env->recovery_altitude_threshold + 100.0f) {
            recovery_complete = 1;  // Gained altitude
        } else if (ticks_in_recovery > 300) {
            recovery_complete = 1;  // Timeout (6 seconds)
        } else if (env->opponent_ap.recovery_phase == 2 && ticks_in_recovery > 150) {
            recovery_complete = 1;  // Been in turn phase for 3 seconds
        }

        if (recovery_complete) {
            env->opponent_recovery_active = 0;
            env->opponent_above_recovery_threshold = 1;  // Ready to detect next crossing
            // Return to level flight
            autopilot_set_mode(&env->opponent_ap, AP_LEVEL, 1.0f, 0.0f, 0.0f);
        }
    } else if (env->selfplay_active && env->recovery_altitude_threshold > 0.0f) {
        // Crossing-based trigger: detect moment when opponent crosses below threshold
        int currently_below = (opp->pos.z < env->recovery_altitude_threshold);
        int was_above = env->opponent_above_recovery_threshold;

        // Detect crossing: was above, now below AND descending
        if (was_above && currently_below && opp->vel.z < 0.0f) {
            // Single probability check at crossing moment
            if (RECOVERY_RAND() < env->recovery_trigger_prob) {
                env->opponent_recovery_active = 1;
                env->opponent_recovery_tick_start = env->tick;
                // Randomize recovery params for this specific recovery
                // Agent learns robust policies against varied opponent behaviors
                float rand_speed = 50.0f + RECOVERY_RAND() * 50.0f;   // 50-100 m/s
                float rand_bank = 30.0f + RECOVERY_RAND() * 45.0f;    // 30-75 degrees
                autopilot_start_recovery(&env->opponent_ap, rand_speed, rand_bank);
                env->log.recovery_triggers += 1.0f;
                if (DEBUG >= 1) printf("[RECOVERY] Triggered at crossing: opp_z=%.0f opp_vz=%.1f tick=%d speed_thr=%.0f bank=%.0f\n",
                       opp->pos.z, opp->vel.z, env->tick, rand_speed, rand_bank);
            }
        }

        // Update tracking state
        env->opponent_above_recovery_threshold = !currently_below;
    }

    // Handle opponent control: recovery takes priority over everything else
    if (env->opponent_recovery_active) {
        // Recovery hijacking active: use autopilot recovery mode
        float opp_actions[5];
        autopilot_step(&env->opponent_ap, &env->opponent, opp_actions, DT);
        for (int i = 0; i < 5; i++) {
            env->last_opp_actions[i] = opp_actions[i];
        }
        step_plane_with_params(&env->opponent, opp_actions, DT, &env->flight_params);
        // No shooting during recovery (disabled in AP_RECOVERY)
    } else if (env->use_opponent_override) {
        // Self-play mode: use externally provided actions from Python
        float opp_actions[5];
        for (int i = 0; i < 5; i++) {
            opp_actions[i] = env->opponent_actions_override[i];
            env->last_opp_actions[i] = opp_actions[i];
        }

        step_plane_with_params(&env->opponent, opp_actions, DT, &env->flight_params);

        // Check if self-play opponent shot the player (two-way combat)
        // Skip if in head-on lockout (guns disabled until pass)
        if (opp_actions[4] > 0.5f && !env->head_on_lockout) {
            // Set fire cooldown for visual tracer effect
            if (env->opponent.fire_cooldown == 0) {
                env->opponent.fire_cooldown = FIRE_COOLDOWN;
            }
            if (check_hit(&env->opponent, &env->player, env->cos_gun_cone)) {
                // Player was shot down by self-play opponent!
                if (DEBUG >= 1) {
                    printf("[SELF-PLAY] Player shot down by opponent policy!\n");
                }
                env->opp_kill = 1;  // Track opponent kill for self-play stats
                env->death_reason = DEATH_KILL;
                env->rewards[0] = -1.0f;
                set_opponent_reward(env, 1.0f);  // Opponent wins (zero-sum)
                env->terminals[0] = 1;
                if (env->debug_log_initialized && env->env_num == 0) debug_log_episode_end(env);
                add_log(env);
                c_reset(env);
                return;
            }
        }
    } else if (env->opponent_ap.mode != AP_STRAIGHT) {
        // Standard autopilot mode (curriculum stages)
        float opp_actions[5];

        // Use AutoAce for stage 20+ (intelligent adversarial opponent)
        if (env->stage >= CURRICULUM_AUTOACE) {
            autoace_step(&env->opponent_ap, &env->opponent_ace,
                        &env->opponent, &env->player, opp_actions, DT);
        } else {
            // Legacy autopilot for curriculum stages 0-19
            env->opponent_ap.threat_pos = env->player.pos;  // For AP_EVASIVE mode
            autopilot_step(&env->opponent_ap, &env->opponent, opp_actions, DT);
        }

        // Store opponent actions for Python access (testing)
        for (int i = 0; i < 5; i++) {
            env->last_opp_actions[i] = opp_actions[i];
        }

        step_plane_with_params(&env->opponent, opp_actions, DT, &env->flight_params);

        // Check if AutoAce shot the player (two-way combat at stage 20+)
        if (env->stage >= CURRICULUM_AUTOACE && opp_actions[4] > 0.5f) {
            if (check_hit(&env->opponent, &env->player, env->cos_gun_cone)) {
                // Player was shot down by AutoAce!
                if (DEBUG >= 1) {
                    printf("[AUTOACE] Player shot down by AutoAce!\n");
                }
                env->opp_kill = 1;  // Track opponent kill for self-play stats
                env->death_reason = DEATH_KILL;  // Reuse KILL (opponent's kill)
                env->rewards[0] = -1.0f;  // Penalty for dying
                set_opponent_reward(env, 1.0f);  // Opponent wins (zero-sum)
                env->terminals[0] = 1;
                if (env->debug_log_initialized && env->env_num == 0) debug_log_episode_end(env);
                add_log(env);
                c_reset(env);
                return;
            }
        }
    } else {
        step_plane(&env->opponent, DT);
    }

    // Track aileron usage for monitoring (no death penalty - see BISECTION.md)
    env->total_aileron_usage += fabsf(env->actions[2]);
    env->aileron_bias += env->actions[2];

#if DEBUG >= 3
    // Track flight envelope diagnostics (only when debugging - expensive)
    {
        Plane *dbg_p = &env->player;
        if (dbg_p->g_force > env->max_g) env->max_g = dbg_p->g_force;
        if (dbg_p->g_force < env->min_g) env->min_g = dbg_p->g_force;
        float speed = norm3(dbg_p->vel);
        if (speed < env->min_speed) env->min_speed = speed;
        if (speed > env->max_speed) env->max_speed = speed;
        if (dbg_p->pos.z < env->min_alt) env->min_alt = dbg_p->pos.z;
        if (dbg_p->pos.z > env->max_alt) env->max_alt = dbg_p->pos.z;
        // Bank angle from quaternion
        float bank = atan2f(2.0f * (dbg_p->ori.w * dbg_p->ori.x + dbg_p->ori.y * dbg_p->ori.z),
                            1.0f - 2.0f * (dbg_p->ori.x * dbg_p->ori.x + dbg_p->ori.y * dbg_p->ori.y));
        if (fabsf(bank) > env->max_bank) env->max_bank = fabsf(bank);
        // Pitch angle from quaternion
        float pitch = asinf(clampf(2.0f * (dbg_p->ori.w * dbg_p->ori.y - dbg_p->ori.z * dbg_p->ori.x), -1.0f, 1.0f));
        if (fabsf(pitch) > env->max_pitch) env->max_pitch = fabsf(pitch);
        // Throttle accumulator
        env->sum_throttle += dbg_p->throttle;
        // Trigger pull edge detection
        int trigger_now = (env->actions[4] > 0.5f) ? 1 : 0;
        if (trigger_now && !env->prev_trigger) env->trigger_pulls++;
        env->prev_trigger = trigger_now;
    }
#endif

    // === Head-on pass detection (for eval mode gun lockout) ===
    if (env->head_on_lockout) {
        // Detect when planes pass each other: dot(rel_pos, rel_vel) flips sign
        Vec3 rel_pos = sub3(env->opponent.pos, env->player.pos);
        Vec3 rel_vel = sub3(env->opponent.vel, env->player.vel);
        float rel_dot = dot3(rel_pos, rel_vel);

        // Sign flip from negative (approaching) to positive (separating) = passed
        if (env->prev_rel_dot < 0 && rel_dot >= 0) {
            env->head_on_lockout = 0;
            if (DEBUG >= 1) {
                fprintf(stderr, "[HEAD-ON] Planes passed - guns unlocked at tick %d\n", env->tick);
            }
        }
        env->prev_rel_dot = rel_dot;
    }

    // === Combat (Phase 5) ===
    Plane *p = &env->player;
    Plane *o = &env->opponent;
    float reward = 0.0f;

    // Decrement fire cooldowns
    // Note: AutoAce (stage 20+) handles opponent cooldown internally in autoace.h
    // Self-play mode also uses opponent cooldown for visual tracer
    if (p->fire_cooldown > 0) p->fire_cooldown--;
    if ((env->use_opponent_override || env->stage < CURRICULUM_AUTOACE) && o->fire_cooldown > 0) o->fire_cooldown--;

    // Player fires: action[4] > 0.5 and cooldown ready and not in head-on lockout
    if (DEBUG >= 10) printf("trigger=%.3f, cooldown=%d, lockout=%d\n", env->actions[4], p->fire_cooldown, env->head_on_lockout);
    if (env->actions[4] > 0.5f && p->fire_cooldown == 0 && !env->head_on_lockout) {
        p->fire_cooldown = FIRE_COOLDOWN;
        env->episode_shots_fired += 1.0f;
        if (DEBUG >= 10) printf("=== FIRED! episode_shots_fired=%.0f ===\n", env->episode_shots_fired);

        // Check if hit = kill = SUCCESS = terminal
        if (check_hit(p, o, env->cos_gun_cone)) {
            if (DEBUG >= 10) printf("*** KILL! ***\n");
            env->kill = 1;
            env->death_reason = DEATH_KILL;
            env->rewards[0] = 1.0f;
            set_opponent_reward(env, -1.0f);  // Opponent loses (zero-sum)
            env->episode_return += 1.0f;
            env->terminals[0] = 1;
            if (env->debug_log_initialized && env->env_num == 0) debug_log_episode_end(env);
            add_log(env);
            c_reset(env);
            return;
        } else {
            if (DEBUG >= 10) printf("MISS (dist=%.1f, in_cone=%d)\n", norm3(sub3(o->pos, p->pos)),
                check_hit(p, o, env->cos_gun_cone));
        }
    }

    // === Reward Shaping (all values from rcfg, sweepable) ===
    Vec3 rel_pos = sub3(o->pos, p->pos);
    float dist = norm3(rel_pos);

    // === df11 Simplified Rewards (6 terms: 3 positive, 3 penalties) ===

    // Calculate timestep-based shaping decay (anneals r_aim and r_closing during self-play)
    // Returns 1.0 before decay window, 0.0 after, linear interpolation between
    float shaping_decay = calc_shaping_decay(
        env->global_step, env->rcfg.shaping_decay_start, env->rcfg.shaping_decay_end);

    // 1. Closing velocity: approaching = good (ANNEALED during self-play)
    Vec3 rel_vel = sub3(p->vel, o->vel);
    Vec3 rel_pos_norm = normalize3(rel_pos);
    float closing_rate = dot3(rel_vel, rel_pos_norm);
    float r_closing = clampf(closing_rate * env->rcfg.closing_scale, -0.05f, 0.05f);
    r_closing *= shaping_decay;  // Anneal during self-play
    reward += r_closing;

    // 2. Aim quality: continuous feedback for gun alignment (ANNEALED during self-play)
    // Shaping rewards teach "how to aim" during curriculum, but become harmful in self-play
    // where they incentivize spiraling rather than killing
    Vec3 player_fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float aim_dot = dot3(rel_pos_norm, player_fwd);  // -1 to +1
    float aim_angle_deg = acosf(clampf(aim_dot, -1.0f, 1.0f)) * RAD_TO_DEG;
    float r_aim = 0.0f;
    if (dist < GUN_RANGE * 2.0f) {  // Only in engagement envelope (~1000m)
        float aim_quality = (aim_dot + 1.0f) * 0.5f;  // Remap [-1,1] to [0,1]
        r_aim = aim_quality * env->rcfg.aim_scale * shaping_decay;  // Anneal during self-play
    }
    reward += r_aim;

    // 3. Negative G penalty: enforce "pull to turn" (realistic)
    float g_threshold = 0.5f;
    float g_deficit = fmaxf(0.0f, g_threshold - p->g_force);
    float r_neg_g = -g_deficit * env->rcfg.neg_g;
    reward += r_neg_g;

    // 4. Stall penalty: speed safety
    float speed = norm3(p->vel);
    float r_stall = 0.0f;
    if (speed < env->rcfg.speed_min) {
        r_stall = -(env->rcfg.speed_min - speed) * PENALTY_STALL;
    }
    reward += r_stall;

    // 5. Rudder penalty: prevent knife-edge climbing (small)
    float r_rudder = -fabsf(env->actions[3]) * PENALTY_RUDDER;
    reward += r_rudder;

    // 5b. Control rate penalty: penalize rapid control changes
    // Sweepable coefficient - find max value that still allows good training
    float d_e = env->actions[1] - env->prev_elevator;
    float d_a = env->actions[2] - env->prev_aileron;
    float d_r = env->actions[3] - env->prev_rudder;
    float delta_sq = d_e*d_e + d_a*d_a + d_r*d_r;
    env->episode_control_rate += delta_sq;  // Always accumulate for logging

    float r_rate = 0.0f;
    if (env->rcfg.control_rate_penalty > 0.0f) {
        r_rate = -delta_sq * env->rcfg.control_rate_penalty;
        reward += r_rate;
    }

    // Update prev actions for next step
    env->prev_elevator = env->actions[1];
    env->prev_aileron = env->actions[2];
    env->prev_rudder = env->actions[3];

    // 6. Progressive altitude penalty: discourage descending rolling scissors
    // Penalty scales quadratically as altitude decreases below threshold
    // Double penalty if also descending - this catches spirals early
    float alt_threshold = env->rcfg.low_altitude_threshold;
    float alt_penalty_scale = env->rcfg.low_altitude_penalty;
    float alt_deficit = fmaxf(0.0f, alt_threshold - p->pos.z);
    float alt_ratio = alt_deficit / fmaxf(alt_threshold, 1.0f);  // 0 at threshold, 1 at 0m
    float r_altitude = -alt_penalty_scale * alt_ratio * alt_ratio;  // Quadratic penalty

    // Double penalty if also descending (catching spirals)
    if (p->vel.z < 0.0f && alt_deficit > 0.0f) {
        float descent_mult = 1.0f + fminf(-p->vel.z / 30.0f, 1.0f);  // Up to 2x at 30m/s descent
        r_altitude *= descent_mult;
    }
    reward += r_altitude;

    // 7. Tiny tick penalty: time preference for faster kills
    float r_time = -0.00001f;
    reward += r_time;

    // 8. Energy management reward: encourage maintaining/gaining energy
    // Asymmetric: +0.001 for gaining energy, -0.0005 for losing (incentivize climbing)
    float player_energy = calc_specific_energy(p);
    float r_player_energy = (player_energy > p->prev_energy) ? 0.001f : -0.0005f;
    reward += r_player_energy;
    p->prev_energy = player_energy;

    // Opponent energy reward (applied to opponent_rewards at end)
    float opp_energy = calc_specific_energy(o);
    float r_opp_energy = (opp_energy > o->prev_energy) ? 0.001f : -0.0005f;
    o->prev_energy = opp_energy;

    // 9. Energy advantage reward: zero-sum reward for relative energy position
    // Encourages staying above opponent (altitude advantage) or faster (speed advantage)
    float energy_diff = player_energy - opp_energy;
    float energy_advantage = clampf(energy_diff / 1000.0f, -1.0f, 1.0f);
    float r_energy_adv = 0.004f * energy_advantage;
    reward += r_energy_adv;
    // Note: opponent gets -r_energy_adv, applied in opponent_rewards section

#if DEBUG >= 2
    // Track aiming diagnostics
    {
        float aim_angle_rad = acosf(clampf(aim_dot, -1.0f, 1.0f));
        if (aim_angle_rad < env->best_aim_angle) env->best_aim_angle = aim_angle_rad;
        if (aim_dot > env->cos_gun_cone) env->ticks_in_cone++;
        if (dist < env->closest_dist) env->closest_dist = dist;
    }
#endif

    // Store per-tick rewards for debug logging
    env->r_closing = r_closing;
    env->r_aim = r_aim;
    env->r_neg_g = r_neg_g;
    env->r_stall = r_stall;
    env->r_rudder = r_rudder;
    env->r_rate = r_rate;
    env->r_altitude = r_altitude;
    env->r_time = r_time;
    env->r_player_energy = r_player_energy;
    env->r_energy_adv = r_energy_adv;

    // Accumulate for episode summary
    env->sum_r_closing += r_closing;
    env->sum_r_aim += r_aim;
    env->sum_r_neg_g += r_neg_g;
    env->sum_r_speed += r_stall;
    env->sum_r_rudder += r_rudder;
    env->sum_r_rate += r_rate;
    env->sum_r_altitude += r_altitude;
    env->sum_r_time += r_time;
    env->sum_r_player_energy += r_player_energy;
    env->sum_r_energy_adv += r_energy_adv;

    if (DEBUG >= 4 && env->env_num == 0) printf("=== REWARD (df11) ===\n");
    if (DEBUG >= 4 && env->env_num == 0) printf("r_closing=%.4f (rate=%.1f m/s)\n", r_closing, closing_rate);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_aim=%.4f (aim_angle=%.1f deg, dist=%.1f)\n", r_aim, aim_angle_deg, dist);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_neg_g=%.5f (g=%.2f)\n", r_neg_g, p->g_force);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_stall=%.4f (speed=%.1f)\n", r_stall, speed);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_rudder=%.5f (rud=%.2f)\n", r_rudder, env->actions[3]);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_rate=%.5f (delta_sq=%.3f)\n", r_rate, delta_sq);
    if (DEBUG >= 4 && env->env_num == 0) printf("reward_total=%.4f\n", reward);

    if (DEBUG >= 10) printf("=== COMBAT ===\n");
    if (DEBUG >= 10) printf("aim_angle=%.1f deg (cone=5 deg)\n", aim_angle_deg);
    if (DEBUG >= 10) printf("dist_to_target=%.1f m (gun_range=500)\n", dist);
    if (DEBUG >= 10) printf("in_cone=%d, in_range=%d\n", aim_dot > env->cos_gun_cone, dist < GUN_RANGE);

    // Global reward clamping to prevent gradient explosion (restored for df8)
    reward = fmaxf(-1.0f, fminf(1.0f, reward));

    env->rewards[0] = reward;
    env->episode_return += reward;

    // Late-training debug logging (only env 0 to reduce volume)
    // Placed AFTER reward calculation so all r_* fields are populated
    if (env->global_step >= env->debug_trigger_step && env->debug_trigger_step > 0) {
        if (!env->debug_log_initialized && env->env_num == 0) {
            init_debug_log(env);
        }
        if (env->env_num == 0) {
            debug_log_tick(env);
        }
    }

    // Check opponent bounds FIRST (opponent crash/OOB = player wins)
    // This handles the "both spiral to ground, one hits first" scenario
    // NOTE: Horizontal bounds removed - real combat has no horizontal walls
    // Only check ground (z < 0) and ceiling (z > WORLD_MAX_Z)
    bool opp_oob = o->pos.z < 0 || o->pos.z > WORLD_MAX_Z;

    if (opp_oob) {
        if (o->pos.z < 0) {
            env->log.opponent_ground_hits += 1.0f;
            if (DEBUG >= 1) printf("[GROUND] Opponent hit ground: z=%.0f tick=%d\n", o->pos.z, env->tick);
        }
        if (DEBUG >= 1) {
            printf("[TERMINAL] Opponent OOB/crashed: pos=(%.1f,%.1f,%.1f)\n",
                   o->pos.x, o->pos.y, o->pos.z);
        }

        // Simplified crash rewards: crasher -1.0, survivor +0.25
        // Not a real kill - opponent crashed on their own
        env->death_reason = DEATH_OOB;
        env->rewards[0] = 0.25f;           // Survivor bonus
        set_opponent_reward(env, -1.0f);   // Crasher penalty
        env->terminals[0] = 1;
        if (env->debug_log_initialized && env->env_num == 0) debug_log_episode_end(env);
        add_log(env);
        c_reset(env);
        return;
    }

    // Check player bounds
    // NOTE: Horizontal bounds removed - real combat has no horizontal walls
    // Only check ground (z < 0) and ceiling (z > WORLD_MAX_Z)
    bool oob = p->pos.z < 0 || p->pos.z > WORLD_MAX_Z;

    // Check for supersonic (physics blowup) - 340 m/s = Mach 1
    float player_speed = norm3(p->vel);
    float opp_speed = norm3(o->vel);
    bool supersonic = player_speed > 340.0f || opp_speed > 340.0f;
    if (DEBUG && supersonic) {
        printf("=== SUPERSONIC BLOWUP ===\n");
        printf("player_speed=%.1f, opp_speed=%.1f\n", player_speed, opp_speed);
        printf("player_vel=(%.1f, %.1f, %.1f)\n", p->vel.x, p->vel.y, p->vel.z);
        printf("opp_vel=(%.1f, %.1f, %.1f)\n", o->vel.x, o->vel.y, o->vel.z);
        printf("opp_ap_mode=%d\n", env->opponent_ap.mode);
    }

    if (oob || env->tick >= env->max_steps || supersonic) {
        if (DEBUG >= 10) printf("=== TERMINAL (FAILURE) ===\n");
        if (DEBUG >= 10) printf("oob=%d, supersonic=%d, tick=%d/%d\n", oob, supersonic, env->tick, env->max_steps);
        // Track death reason (priority: supersonic > oob > timeout)
        if (supersonic) {
            env->death_reason = DEATH_SUPERSONIC;
            // Physics blowup - both policies penalized equally
            env->rewards[0] = -1.0f;
            set_opponent_reward(env, -1.0f);
        } else if (oob) {
            if (p->pos.z < 0) {
                env->log.player_ground_hits += 1.0f;
                if (DEBUG >= 1) printf("[GROUND] Player hit ground: z=%.0f tick=%d\n", p->pos.z, env->tick);
            }
            // Simplified crash rewards: crasher -1.0, survivor +0.25
            env->death_reason = DEATH_OOB;
            env->rewards[0] = -1.0f;           // Crasher penalty
            set_opponent_reward(env, 0.25f);   // Survivor bonus
        } else {
            // Timeout - both failed to achieve kill
            env->death_reason = DEATH_TIMEOUT;
            env->rewards[0] = -0.5f;
            set_opponent_reward(env, -0.5f);
        }
        env->terminals[0] = 1;
        if (env->debug_log_initialized && env->env_num == 0) debug_log_episode_end(env);
        add_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
#if DEBUG >= 5
    print_observations(env);
#endif

    // Compute opponent observations and rewards (for dual self-play with Multiprocessing)
    // Only if buffers are provided by Python (non-NULL)
    if (env->opponent_observations != NULL) {
        compute_opponent_observations(env, env->opponent_observations);
    }
    if (env->opponent_rewards != NULL) {
        // Zero-sum game: opponent reward = negative of player reward
        // PLUS independent penalties/rewards (not zero-sum)
        float opp_reward = -env->rewards[0];

        // Progressive altitude penalty (same calculation as player)
        float opp_alt_deficit = fmaxf(0.0f, alt_threshold - o->pos.z);
        float opp_alt_ratio = opp_alt_deficit / fmaxf(alt_threshold, 1.0f);
        float opp_r_altitude = -alt_penalty_scale * opp_alt_ratio * opp_alt_ratio;
        if (o->vel.z < 0.0f && opp_alt_deficit > 0.0f) {
            float opp_descent_mult = 1.0f + fminf(-o->vel.z / 30.0f, 1.0f);
            opp_r_altitude *= opp_descent_mult;
        }
        opp_reward += opp_r_altitude;

        // Energy management reward (calculated above in section 8)
        opp_reward += r_opp_energy;

        // Energy advantage (zero-sum: opponent gets negative of player's)
        opp_reward += -r_energy_adv;

        env->opponent_rewards[0] = opp_reward;
    }
}

void c_close(Dogfight *env);

#include "dogfight_render.h"

// Force exact game state for testing. Defaults shown in comments are applied in Python.
void force_state(
    Dogfight *env,
    float p_px,        // = 0.0f, player pos X
    float p_py,        // = 0.0f, player pos Y
    float p_pz,        // = 1000.0f, player pos Z
    float p_vx,        // = 150.0f, player vel X (m/s)
    float p_vy,        // = 0.0f, player vel Y
    float p_vz,        // = 0.0f, player vel Z
    float p_ow,        // = 1.0f, player orientation quat W
    float p_ox,        // = 0.0f, player orientation quat X
    float p_oy,        // = 0.0f, player orientation quat Y
    float p_oz,        // = 0.0f, player orientation quat Z
    float p_throttle,  // = 1.0f, player throttle [0,1]
    float o_px,        // = -9999.0f (auto: 400m ahead), opponent pos X
    float o_py,        // = -9999.0f (auto), opponent pos Y
    float o_pz,        // = -9999.0f (auto), opponent pos Z
    float o_vx,        // = -9999.0f (auto: match player), opponent vel X
    float o_vy,        // = -9999.0f (auto), opponent vel Y
    float o_vz,        // = -9999.0f (auto), opponent vel Z
    float o_ow,        // = -9999.0f (auto: match player), opponent ori W
    float o_ox,        // = -9999.0f (auto), opponent ori X
    float o_oy,        // = -9999.0f (auto), opponent ori Y
    float o_oz,        // = -9999.0f (auto), opponent ori Z
    int tick,          // = 0, environment tick
    int p_cooldown,    // = -1 (no change), player fire cooldown ticks
    int o_cooldown     // = -1 (no change), opponent fire cooldown ticks
) {
    env->player.pos = vec3(p_px, p_py, p_pz);
    env->player.vel = vec3(p_vx, p_vy, p_vz);
    env->player.prev_vel = vec3(p_vx, p_vy, p_vz);  // Initialize to current (no accel)
    env->player.omega = vec3(0, 0, 0);  // No angular velocity
    env->player.ori = quat(p_ow, p_ox, p_oy, p_oz);
    quat_normalize(&env->player.ori);
    env->player.throttle = p_throttle;
    env->player.fire_cooldown = (p_cooldown >= 0) ? p_cooldown : 0;
    env->player.yaw_from_rudder = 0.0f;

    // Opponent position: auto = 400m ahead of player
    if (o_px < -9000.0f) {
        Vec3 fwd = quat_rotate(env->player.ori, vec3(1, 0, 0));
        env->opponent.pos = add3(env->player.pos, mul3(fwd, 400.0f));
    } else {
        env->opponent.pos = vec3(o_px, o_py, o_pz);
    }

    // Opponent velocity: auto = match player
    if (o_vx < -9000.0f) {
        env->opponent.vel = env->player.vel;
    } else {
        env->opponent.vel = vec3(o_vx, o_vy, o_vz);
    }

    // Opponent orientation: auto = match player
    if (o_ow < -9000.0f) {
        env->opponent.ori = env->player.ori;
    } else {
        env->opponent.ori = quat(o_ow, o_ox, o_oy, o_oz);
        quat_normalize(&env->opponent.ori);
    }
    env->opponent.fire_cooldown = (o_cooldown >= 0) ? o_cooldown : 0;
    env->opponent.yaw_from_rudder = 0.0f;
    env->opponent.prev_vel = env->opponent.vel;  // Initialize to current (no accel)
    env->opponent.omega = vec3(0, 0, 0);  // No angular velocity

    // Reset autopilot PID state to avoid derivative spikes
    env->opponent_ap.prev_vz = env->opponent.vel.z;
    env->opponent_ap.prev_bank_error = 0.0f;

    // Environment state
    env->tick = tick;
    env->episode_return = 0.0f;

    compute_observations(env);
#if DEBUG >= 5
    print_observations(env);
#endif
}
