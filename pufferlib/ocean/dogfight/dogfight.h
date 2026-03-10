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
    // OBS_MOMENTUM_GFORCE was scheme 0 (17 obs) — removed, code preserved in dogfight_observations.h
    OBS_PILOT = 0,              // Pilot awareness (22 obs) — was scheme 1
    OBS_OPPONENT_AWARE = 1,     // Pilot + opp up vector + opp speed (26 obs) — was scheme 2
    OBS_SCHEME_COUNT
} ObsScheme;

static const int OBS_SIZES[OBS_SCHEME_COUNT] = {22, 26};

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
// Forced vertical merge spawns (self-play curriculum)
static void spawn_vertical_apex(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_vertical_past(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_vertical_midclimb(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_vertical_merge(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);
static void spawn_vertical_premerge(struct Dogfight *env, Vec3 player_pos, Vec3 player_vel);

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
    // Energy management rewards (sweepable)
    float energy_gain_scale;       // Reward for gaining energy (default 0.001)
    float energy_loss_scale;       // Penalty for losing energy (default 0.0005)
    float energy_advantage_scale;  // Zero-sum energy advantage scale (default 0.004)
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
    // Opponent observation scheme (for cross-scheme evaluation)
    // -1 = use same as player (default), >=0 = separate scheme for opponent
    int opponent_obs_scheme;
    int opponent_obs_size;
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
    unsigned char obs_highlight[32];  // 1 = highlight this observation with red arrow (max scheme is 27 obs)
    // Last opponent actions (for Python access in tests)
    float last_opp_actions[5];  // throttle, elevator, aileron, rudder, trigger
    // Camera control
    int camera_follow_opponent;  // 0 = follow player (default), 1 = follow opponent
    // Self-play: external opponent actions override (Phase 1)
    float opponent_actions_override[5];  // [throttle, elevator, aileron, rudder, trigger]
    int use_opponent_override;           // 0 = use autopilot, 1 = use override
    float selfplay_prob;                 // 0.0=all autopilot, 1.0=all neural (per-episode dice roll)
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

    // Forced vertical merge curriculum (teaches vertical fighting via spawn geometry)
    float vertical_spawn_prob;    // Probability of forced vertical spawn during self-play (0.0-1.0)
    int vertical_level;           // Vertical sub-level: 0=apex, 1=past-vertical, 2=mid-climb, 3=merge, 4=pre-merge
    int vertical_spawn_used;      // 1 if vertical spawn was triggered this reset (skip speed randomization)

    // Previous values for rate observations (kept for preserved rate functions)
    // Player perspective
    float prev_player_target_az;
    float prev_player_target_el;
    float prev_player_aspect;
    float prev_player_eadv;
    // Opponent perspective (for self-play)
    float prev_opp_target_az;
    float prev_opp_target_el;
    float prev_opp_aspect;
    float prev_opp_eadv;

    // Runtime-configurable flight physics (for parameter sweeps + domain randomization)
    FlightParams flight_params;
    float domain_randomization;  // 0.0 = off, 0.1 = +/-10% per-episode variation
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
    // Opponent obs scheme defaults to same as player (-1 = inherit)
    env->opponent_obs_scheme = -1;
    env->opponent_obs_size = env->obs_size;
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
    env->selfplay_prob = 1.0f;  // Default: all neural when override enabled
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

    // Forced vertical merge curriculum: disabled by default, enabled by Python
    env->vertical_spawn_prob = 0.0f;
    env->vertical_level = 0;
    env->vertical_spawn_used = 0;

    // Rate observation previous values (schemes 4, 5)
    env->prev_player_target_az = 0.0f;
    env->prev_player_target_el = 0.0f;
    env->prev_player_aspect = 0.0f;
    env->prev_player_eadv = 0.0f;
    env->prev_opp_target_az = 0.0f;
    env->prev_opp_target_el = 0.0f;
    env->prev_opp_aspect = 0.0f;
    env->prev_opp_eadv = 0.0f;
}

void set_obs_highlight(Dogfight *env, int *indices, int count) {
    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));
    for (int i = 0; i < count && i < 32; i++) {
        if (indices[i] >= 0 && indices[i] < 32) {
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

#include "dogfight_spawn.h"

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

    // Reset rate observation previous values (schemes 4, 5)
    env->prev_player_target_az = 0.0f;
    env->prev_player_target_el = 0.0f;
    env->prev_player_aspect = 0.0f;
    env->prev_player_eadv = 0.0f;
    env->prev_opp_target_az = 0.0f;
    env->prev_opp_target_el = 0.0f;
    env->prev_opp_aspect = 0.0f;
    env->prev_opp_eadv = 0.0f;

    // Gun cone for hit detection - stays fixed at 5°
    env->cos_gun_cone = cosf(env->gun_cone_angle);

    // Domain randomization: randomize physics params per-episode
    randomize_flight_params(&env->flight_params, env->domain_randomization);

    // Spawn player at random position with base velocity
    // Use most of the sky (800-4200m) but avoid very low altitudes
    Vec3 pos = vec3(rndf(-500, 500), rndf(-500, 500), rndf(800, 4200));
    Vec3 vel = vec3(80, 0, 0);  // Base speed, will be randomized below
    reset_plane(&env->player, pos, vel);

    // Spawn opponent based on curriculum stage (or legacy if disabled)
    if (env->curriculum_enabled) {
        spawn_by_curriculum(env, pos, vel);

        // Phase 1: Apply stage-dependent speed randomization to both planes
        // Skip if vertical spawn was used (it sets specific speeds for energy state)
        if (!env->vertical_spawn_used) {
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
        }
    } else {
        spawn_legacy(env, pos, vel);
    }

    if (DEBUG >= 10) printf("=== RESET ===\n");
    if (DEBUG >= 10) printf("kill=%d, episode_shots_fired=%.0f (now cleared)\n", env->kill, env->episode_shots_fired);
    if (DEBUG >= 10) printf("player_pos=(%.1f, %.1f, %.1f)\n", pos.x, pos.y, pos.z);
    if (DEBUG >= 10) printf("player_vel=(%.1f, %.1f, %.1f) speed=%.1f\n", vel.x, vel.y, vel.z, norm3(vel));
    if (DEBUG >= 10) printf("opponent_pos=(%.1f, %.1f, %.1f)\n", env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z);
    if (DEBUG >= 10) printf("initial_dist=%.1f m, stage=%d\n", norm3(sub3(env->opponent.pos, pos)), env->stage);

    // Per-episode: probabilistically choose neural vs autopilot opponent
    if (env->selfplay_active) {
        if (rndf(0, 1) < env->selfplay_prob) {
            env->use_opponent_override = 1;  // Neural opponent this episode
        } else {
            env->use_opponent_override = 0;  // Autopilot (autoace) this episode
        }
    }

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
    // Asymmetric: gain_scale for gaining energy, -loss_scale for losing (incentivize climbing)
    float player_energy = calc_specific_energy_with_params(p, &env->flight_params);
    float r_player_energy = (player_energy > p->prev_energy)
        ? env->rcfg.energy_gain_scale : -env->rcfg.energy_loss_scale;
    reward += r_player_energy;
    p->prev_energy = player_energy;

    // Opponent energy reward (applied to opponent_rewards at end)
    float opp_energy = calc_specific_energy_with_params(o, &env->flight_params);
    float r_opp_energy = (opp_energy > o->prev_energy)
        ? env->rcfg.energy_gain_scale : -env->rcfg.energy_loss_scale;
    o->prev_energy = opp_energy;

    // 9. Energy advantage reward: zero-sum reward for relative energy position
    // Encourages staying above opponent (altitude advantage) or faster (speed advantage)
    float energy_diff = player_energy - opp_energy;
    float energy_advantage = clampf(energy_diff / 1000.0f, -1.0f, 1.0f);
    float r_energy_adv = env->rcfg.energy_advantage_scale * energy_advantage;
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
