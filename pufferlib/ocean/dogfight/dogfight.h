// dogfight.h - WW2 aerial combat environment
// Uses flightlib.h for flight physics

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"
#include "rlgl.h"  // For rlSetClipPlanes()

// Define DEBUG before including flightlib.h so physics functions can use it
#define DEBUG 0

#include "flightlib.h"
#include "autopilot.h"

// Observation scheme enumeration
typedef enum {
    OBS_ANGLES = 0,              // Spherical coordinates (12 obs)
    OBS_PURSUIT = 1,             // Energy-aware pursuit observations (13 obs)
    OBS_REALISTIC = 2,           // Cockpit instruments only (10 obs)
    OBS_REALISTIC_RANGE = 3,     // REALISTIC with explicit range (10 obs)
    OBS_REALISTIC_ENEMY_STATE = 4, // + enemy pitch/roll/heading (13 obs)
    OBS_REALISTIC_FULL = 5,      // + turn rate + G-loading (15 obs)
    OBS_SCHEME_COUNT
} ObsScheme;

// Observation size lookup table
static const int OBS_SIZES[OBS_SCHEME_COUNT] = {12, 13, 10, 10, 13, 15};

// Observation labels for each scheme (for HUD display)
// Scheme 0: OBS_ANGLES (12 obs)
static const char* OBS_LABELS_ANGLES[12] = {
    "px", "py", "pz", "speed", "pitch", "roll", "yaw",
    "tgt_az", "tgt_el", "dist", "closure", "opp_hdg"
};

// Scheme 1: OBS_PURSUIT (13 obs)
static const char* OBS_LABELS_PURSUIT[13] = {
    "speed", "potential", "pitch", "roll", "energy",
    "tgt_az", "tgt_el", "dist", "closure",
    "tgt_roll", "tgt_pitch", "aspect", "E_adv"
};

// Scheme 2: OBS_REALISTIC (10 obs)
static const char* OBS_LABELS_REALISTIC[10] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "tgt_size",
    "aspect", "horizon", "dist"
};

// Scheme 3: OBS_REALISTIC_RANGE (10 obs)
static const char* OBS_LABELS_REALISTIC_RANGE[10] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "range_km",
    "aspect", "horizon", "closure"
};

// Scheme 4: OBS_REALISTIC_ENEMY_STATE (13 obs)
static const char* OBS_LABELS_REALISTIC_ENEMY_STATE[13] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "range_km",
    "aspect", "horizon", "closure",
    "emy_pitch", "emy_roll", "emy_hdg"
};

// Scheme 5: OBS_REALISTIC_FULL (15 obs)
static const char* OBS_LABELS_REALISTIC_FULL[15] = {
    "airspeed", "altitude", "pitch", "roll",
    "tgt_az", "tgt_el", "range_km",
    "aspect", "horizon", "closure",
    "emy_pitch", "emy_roll", "emy_hdg",
    "turn_rate", "g_load"
};

// Curriculum learning stages (progressive difficulty)
// Reordered 2026-01-18: moved CROSSING from stage 2 to stage 6 (see CURRICULUM_PLANS.md)
typedef enum {
    CURRICULUM_TAIL_CHASE = 0,   // Easiest: opponent ahead, same heading
    CURRICULUM_HEAD_ON,          // Opponent coming toward us
    CURRICULUM_VERTICAL,         // Above or below player (was stage 3)
    CURRICULUM_MANEUVERING,      // Opponent does turns (was stage 4)
    CURRICULUM_FULL_RANDOM,      // Mix of all basic modes (was stage 5)
    CURRICULUM_HARD_MANEUVERING, // Hard turns + weave patterns (was stage 6)
    CURRICULUM_CROSSING,         // 45 degree deflection shots (was stage 2, reduced from 90°)
    CURRICULUM_EVASIVE,          // Reactive evasion (hardest)
    CURRICULUM_COUNT
} CurriculumStage;

// Stage difficulty weights for composite metric (higher = harder = more valuable)
// Used to compute difficulty_weighted_perf = perf * avg_stage_weight
// Reordered 2026-01-18 to match new enum order (see CURRICULUM_PLANS.md)
static const float STAGE_WEIGHTS[CURRICULUM_COUNT] = {
    0.2f,   // TAIL_CHASE - trivial
    0.3f,   // HEAD_ON - easy
    0.4f,   // VERTICAL - medium (was stage 3)
    0.5f,   // MANEUVERING - medium (was stage 4)
    0.65f,  // FULL_RANDOM - medium-hard (was stage 5)
    0.8f,   // HARD_MANEUVERING - hard (was stage 6)
    0.9f,   // CROSSING - hard, 45° deflection (was stage 2)
    1.0f    // EVASIVE - hardest
};

// Simulation timing
#define DT 0.02f

// World bounds
#define WORLD_HALF_X 2000.0f
#define WORLD_HALF_Y 2000.0f
#define WORLD_MAX_Z 3000.0f
#define MAX_SPEED 250.0f
#define OBS_SIZE 19  // player(13) + rel_pos(3) + rel_vel(3)

// Inverse constants for faster normalization (multiply instead of divide)
#define INV_WORLD_HALF_X 0.0005f       // 1/2000
#define INV_WORLD_HALF_Y 0.0005f       // 1/2000
#define INV_WORLD_MAX_Z  0.000333333f  // 1/3000
#define INV_MAX_SPEED    0.004f        // 1/250
#define INV_PI           0.31830988618f // 1/PI
#define INV_HALF_PI      0.63661977236f // 2/PI (i.e., 1/(PI*0.5))

// Combat constants
#define GUN_RANGE 500.0f       // meters
#define INV_GUN_RANGE 0.002f   // 1/500
#define GUN_CONE_ANGLE 0.087f  // ~5 degrees in radians
#define FIRE_COOLDOWN 10       // ticks (0.2 seconds at 50Hz)

typedef struct Log {
    float episode_return;
    float episode_length;
    float score;           // 1.0 on kill, 0.0 on failure
    float perf;            // sweep metric (same as kills)
    float kills;           // cumulative kills
    float shots_fired;     // cumulative shots
    float accuracy;        // kills / shots_fired * 100
    float stage;           // current curriculum stage (for monitoring)
    // Curriculum-weighted metrics (Phase 1)
    float total_stage_weight;       // Sum of stage weights across all episodes
    float avg_stage_weight;         // total_stage_weight / n
    float total_abs_bias;           // Sum of |aileron_bias| at episode end
    float avg_abs_bias;             // total_abs_bias / n
    float ultimate;                 // Main sweep metric: kill_rate * avg_stage_weight / (1 + avg_abs_bias * 0.01)
    float n;
} Log;

// Death reason tracking for diagnostics
typedef enum DeathReason {
    DEATH_NONE = 0,      // Episode still running
    DEATH_KILL = 1,      // Player scored a kill (success)
    DEATH_OOB = 2,       // Out of bounds
    DEATH_AILERON = 3,   // Aileron limit exceeded
    DEATH_TIMEOUT = 4,   // Max steps reached
    DEATH_SUPERSONIC = 5 // Physics blowup
} DeathReason;

// Reward configuration (df11: simplified - 6 terms instead of 9+)
typedef struct RewardConfig {
    // Positive shaping
    float aim_scale;         // Continuous aiming reward (default 0.05)
    float closing_scale;     // +N per m/s closing (default 0.003)
    // Penalties
    float neg_g;             // -N per unit G below 0.5 (default 0.02) - enforces "pull to turn"
    float stall;             // -N per m/s below speed_min (default 0.002)
    float rudder;            // -N per unit rudder magnitude (default 0.001) - prevents knife-edge
    // Thresholds
    float speed_min;         // Stall threshold (default 50.0)
} RewardConfig;

typedef struct Client {
    Camera3D camera;
    float width;
    float height;
    // Camera orbit state (for mouse control)
    float cam_distance;
    float cam_azimuth;
    float cam_elevation;
    bool is_dragging;
    float last_mouse_x;
    float last_mouse_y;
} Client;

typedef struct Dogfight {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;
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
    float cos_gun_cone_2x;  // cosf(gun_cone_angle * 2)
    // Reward shaping cone (anneals from large to small)
    float reward_cone_angle;   // Current reward cone (radians) - anneals
    float cos_reward_cone;     // cosf(reward_cone_angle)
    float cos_reward_cone_2x;  // cosf(reward_cone_angle * 2)
    // Aim cone annealing parameters
    float aim_cone_start;      // Starting reward cone (radians, e.g., 20° = 0.35)
    float aim_cone_end;        // Ending reward cone (radians, e.g., 5° = 0.087)
    int aim_anneal_episodes;   // Episodes to fully anneal
    // Opponent autopilot
    AutopilotState opponent_ap;
    // Observation scheme
    int obs_scheme;
    int obs_size;
    // Reward configuration (sweepable)
    RewardConfig rcfg;
    // Episode-level tracking (reset each episode)
    int kill;                   // 1 if killed this episode, 0 otherwise
    float episode_shots_fired;  // For accuracy tracking
    // Curriculum learning
    int curriculum_enabled;     // 0 = off (legacy spawning), 1 = on
    int curriculum_randomize;   // 0 = progressive (training), 1 = random stage each episode (eval)
    int total_episodes;         // Cumulative episodes (persists across resets)
    CurriculumStage stage;      // Current difficulty stage
    int is_initialized;         // Flag to preserve curriculum state across re-init (for Multiprocessing)
    // Performance-based curriculum
    float recent_kills;         // Kills in current evaluation window
    float recent_episodes;      // Episodes in current evaluation window
    float advance_threshold;    // Kill rate to advance (default 0.7)
    float demote_threshold;     // Kill rate to demote (default 0.3)
    int eval_window;            // Episodes per evaluation (default 50)
    // Anti-spinning
    float total_aileron_usage;  // Accumulated |aileron| input (for spin death)
    float aileron_bias;         // Cumulative signed aileron (for directional penalty)
    // Episode reward accumulators (for DEBUG summaries)
    float sum_r_closing;
    float sum_r_speed;      // Stall penalty
    float sum_r_neg_g;
    float sum_r_rudder;
    float sum_r_aim;
    // Aiming diagnostics (reset each episode, for DEBUG output)
    float best_aim_angle;    // Best (smallest) aim angle achieved (radians)
    int ticks_in_cone;       // Ticks where aim_dot > cos_reward_cone
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
    // Debug
    int env_num;                // Environment index (for filtering debug output)
    // Observation highlighting (for visual debugging)
    unsigned char obs_highlight[16];  // 1 = highlight this observation with red arrow
} Dogfight;

void init(Dogfight *env, int obs_scheme, RewardConfig *rcfg, int curriculum_enabled, int curriculum_randomize, float aim_cone_start, float aim_cone_end, int aim_anneal_episodes, float advance_threshold, float demote_threshold, int eval_window, int env_num) {
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
    env->cos_gun_cone_2x = cosf(env->gun_cone_angle * 2.0f);
    // Aim cone annealing for REWARD SHAPING
    env->aim_cone_start = aim_cone_start > 0.0f ? aim_cone_start : 0.35f;  // Default 20°
    env->aim_cone_end = aim_cone_end > 0.0f ? aim_cone_end : GUN_CONE_ANGLE;  // Default 5°
    env->aim_anneal_episodes = aim_anneal_episodes > 0 ? aim_anneal_episodes : 50000;
    // Initialize reward cone to start value
    env->reward_cone_angle = env->aim_cone_start;
    env->cos_reward_cone = cosf(env->reward_cone_angle);
    env->cos_reward_cone_2x = cosf(env->reward_cone_angle * 2.0f);
    // Initialize opponent autopilot
    autopilot_init(&env->opponent_ap);
    // Reward configuration (copy from provided config)
    env->rcfg = *rcfg;
    // Episode tracking
    env->kill = 0;
    env->episode_shots_fired = 0.0f;
    // Curriculum learning
    env->curriculum_enabled = curriculum_enabled;
    env->curriculum_randomize = curriculum_randomize;
    // Only reset curriculum state on first init (preserve across re-init for Multiprocessing)
    if (!env->is_initialized) {
        env->total_episodes = 0;
        env->stage = CURRICULUM_TAIL_CHASE;
        // Performance-based curriculum
        env->recent_kills = 0.0f;
        env->recent_episodes = 0.0f;
        env->advance_threshold = advance_threshold > 0.0f ? advance_threshold : 0.7f;
        env->demote_threshold = demote_threshold > 0.0f ? demote_threshold : 0.3f;
        env->eval_window = eval_window > 0 ? eval_window : 50;
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
    // Clear observation highlights
    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));
}

// Set which observations to highlight with red arrows (for visual debugging)
// indices: array of observation indices to highlight
// count: number of indices
void set_obs_highlight(Dogfight *env, int *indices, int count) {
    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));
    for (int i = 0; i < count && i < 16; i++) {
        if (indices[i] >= 0 && indices[i] < 16) {
            env->obs_highlight[indices[i]] = 1;
        }
    }
}

void add_log(Dogfight *env) {
    // Level 1: Episode summary (one line, easy to grep)
    if (DEBUG >= 1 && env->env_num == 0) {
        const char* death_names[] = {"NONE", "KILL", "OOB", "AILERON", "TIMEOUT", "SUPERSONIC"};
        float mean_ail = env->total_aileron_usage / fmaxf((float)env->tick, 1.0f);
        printf("EP tick=%d ret=%.2f death=%s kill=%d stage=%d total_eps=%d mean_ail=%.2f bias=%.1f\n",
               env->tick, env->episode_return, death_names[env->death_reason],
               env->kill, env->stage, env->total_episodes, mean_ail, env->aileron_bias);
    }

    // Level 2: Reward breakdown (which components dominated?)
    if (DEBUG >= 2 && env->env_num == 0) {
        printf("  SHAPING: closing=%+.2f aim=%+.2f\n", env->sum_r_closing, env->sum_r_aim);
        printf("  PENALTY: stall=%.2f neg_g=%.2f rudder=%.2f\n",
               env->sum_r_speed, env->sum_r_neg_g, env->sum_r_rudder);
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
    env->log.kills += env->kill ? 1.0f : 0.0f;
    env->log.score += env->rewards[0];
    env->log.shots_fired += env->episode_shots_fired;
    env->log.accuracy = (env->log.shots_fired > 0.0f) ? (env->log.kills / env->log.shots_fired * 100.0f) : 0.0f;
    env->log.stage = (float)env->stage;  // Track curriculum stage

    // Curriculum-weighted metrics (Phase 1)
    // Track difficulty faced and compute composite metric
    env->log.total_stage_weight += STAGE_WEIGHTS[env->stage];
    env->log.total_abs_bias += fabsf(env->aileron_bias);  // Track bias at episode end
    env->log.n += 1.0f;
    env->log.avg_stage_weight = env->log.total_stage_weight / env->log.n;
    env->log.avg_abs_bias = env->log.total_abs_bias / env->log.n;

    // ultimate = kill_rate * stage_weight / (1 + avg_abs_bias * 0.01)
    // Rewards killing hard opponents, penalizes degenerate aileron bias
    float kill_rate = env->log.kills / env->log.n;
    float difficulty_weighted = kill_rate * env->log.avg_stage_weight;
    float bias_divisor = 1.0f + env->log.avg_abs_bias * 0.1f;  // min 1.0, safe
    env->log.ultimate = difficulty_weighted / bias_divisor;

    if (DEBUG >= 10) printf("  log.perf=%.2f, log.shots_fired=%.0f, log.n=%.0f\n", env->log.perf, env->log.shots_fired, env->log.n);

    if (env->curriculum_enabled && !env->curriculum_randomize) {
        env->recent_episodes += 1.0f;
        env->recent_kills += env->kill ? 1.0f : 0.0f;

        // Evaluate every eval_window episodes
        if (env->recent_episodes >= (float)env->eval_window) {
            float recent_rate = env->recent_kills / env->recent_episodes;

            if (recent_rate > env->advance_threshold && env->stage < CURRICULUM_COUNT - 1) {
                env->stage++;
                if (DEBUG >= 1) {
                    fprintf(stderr, "[ADVANCE] env=%d stage->%d (rate=%.2f, window=%d)\n",
                            env->env_num, env->stage, recent_rate, env->eval_window);
                }
            } else if (recent_rate < env->demote_threshold && env->stage > 0) {
                env->stage--;
                if (DEBUG >= 1) {
                    fprintf(stderr, "[DEMOTE] env=%d stage->%d (rate=%.2f, window=%d)\n",
                            env->env_num, env->stage, recent_rate, env->eval_window);
                }
            }

            env->recent_kills = 0.0f;
            env->recent_episodes = 0.0f;
        }
    }
}

// Scheme 0: Angles observations (spherical coordinates)
void compute_obs_angles(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles from quaternion
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));
    float yaw = atan2f(2.0f * (p->ori.w * p->ori.z + p->ori.x * p->ori.y),
                       1.0f - 2.0f * (p->ori.y * p->ori.y + p->ori.z * p->ori.z));

    // Target in body frame → spherical
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float azimuth = atan2f(rel_pos_body.y, rel_pos_body.x);  // -pi to pi
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float elevation = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));  // -pi/2 to pi/2

    // Closing rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closing_rate = dot3(rel_vel, normalize3(rel_pos));

    // Opponent heading relative to player
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 opp_fwd_body = quat_rotate(q_inv, opp_fwd);
    float opp_heading = atan2f(opp_fwd_body.y, opp_fwd_body.x);

    int i = 0;
    // Player state
    env->observations[i++] = p->pos.x * INV_WORLD_HALF_X;
    env->observations[i++] = p->pos.y * INV_WORLD_HALF_Y;
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);  // Speed scalar
    env->observations[i++] = pitch * INV_PI;      // -0.5 to 0.5
    env->observations[i++] = roll * INV_PI;       // -1 to 1
    env->observations[i++] = yaw * INV_PI;        // -1 to 1

    // Target angles
    env->observations[i++] = azimuth * INV_PI;    // -1 to 1
    env->observations[i++] = elevation * INV_HALF_PI;  // -1 to 1
    env->observations[i++] = clampf(dist * INV_GUN_RANGE, 0.0f, 2.0f) - 1.0f;  // [-1,1]
    env->observations[i++] = clampf(closing_rate * INV_MAX_SPEED, -1.0f, 1.0f);  // Clamped to [-1,1]

    // Opponent info
    env->observations[i++] = opp_heading * INV_PI;  // -1 to 1
    // OBS_SIZE = 12
}

// Scheme 1: OBS_PURSUIT - Energy-aware pursuit observations (13 obs)
// Better than old OBS_CONTROL_ERROR: no spoon-feeding of control errors,
// instead provides body-frame target info and energy state for learning pursuit
void compute_obs_pursuit(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Own Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Own energy state: (potential + kinetic) / 2, normalized to [0,1]
    float speed = norm3(p->vel);
    float alt = p->pos.z;
    float potential = alt * INV_WORLD_MAX_Z;
    float kinetic = (speed * speed) / (MAX_SPEED * MAX_SPEED);
    float own_energy = (potential + kinetic) * 0.5f;

    // Target in body frame
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure = dot3(rel_vel, normalize3(rel_pos));

    // Target Euler angles
    float target_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float target_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                               1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));

    // Target aspect (head-on vs tail)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Target energy
    float opp_speed = norm3(o->vel);
    float opp_alt = o->pos.z;
    float opp_potential = opp_alt * INV_WORLD_MAX_Z;
    float opp_kinetic = (opp_speed * opp_speed) / (MAX_SPEED * MAX_SPEED);
    float opp_energy = (opp_potential + opp_kinetic) * 0.5f;

    // Energy advantage
    float energy_advantage = clampf(own_energy - opp_energy, -1.0f, 1.0f);

    int i = 0;
    // Own flight state (5 obs)
    env->observations[i++] = clampf(speed * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = potential;
    env->observations[i++] = pitch * INV_HALF_PI;
    env->observations[i++] = roll * INV_PI;
    env->observations[i++] = own_energy;

    // Target position in body frame (4 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = clampf(dist * INV_GUN_RANGE, 0.0f, 2.0f) - 1.0f;
    env->observations[i++] = clampf(closure * INV_MAX_SPEED, -1.0f, 1.0f);

    // Target state (3 obs)
    env->observations[i++] = target_roll * INV_PI;
    env->observations[i++] = target_pitch * INV_HALF_PI;
    env->observations[i++] = target_aspect;

    // Energy comparison (1 obs)
    env->observations[i++] = energy_advantage;
    // OBS_SIZE = 13
}

// Scheme 4: Realistic cockpit instruments only
void compute_obs_realistic(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Target apparent size (larger when closer)
    float target_size = 20.0f / fmaxf(dist, 10.0f);  // ~wingspan/distance

    // Opponent aspect (are they facing toward/away from us?)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);  // 1 = head-on, -1 = tail

    // Horizon visible (is up vector pointing up?)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;  // 1 = level, 0 = knife-edge, -1 = inverted

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);  // Airspeed
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;     // Altitude
    env->observations[i++] = pitch * INV_HALF_PI;            // Pitch indicator
    env->observations[i++] = roll * INV_PI;                       // Bank indicator

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;                  // Target azimuth in sight
    env->observations[i++] = target_el * INV_HALF_PI;         // Target elevation in sight
    env->observations[i++] = clampf(target_size, 0.0f, 2.0f) - 1.0f;  // Target size

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;                   // -1 to 1
    env->observations[i++] = horizon_visible;                 // -1 to 1
    env->observations[i++] = clampf(dist * INV_GUN_RANGE, 0.0f, 2.0f) - 1.0f;  // Distance estimate
    // OBS_SIZE = 10
}

// Scheme 3: REALISTIC with explicit range (10 obs)
// Like REALISTIC but with km range + closure rate instead of target_size + distance_estimate
void compute_obs_realistic_range(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Range in km (0 = point blank, 0.5 = 1km, 1.0 = 2km+)
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect (are they facing toward/away from us?)
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);  // 1 = head-on, -1 = tail

    // Horizon visible (is up vector pointing up?)
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;  // 1 = level, 0 = knife-edge, -1 = inverted

    // Closure rate (positive = closing)
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);  // Airspeed
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;     // Altitude
    env->observations[i++] = pitch * INV_HALF_PI;            // Pitch indicator
    env->observations[i++] = roll * INV_PI;                       // Bank indicator

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;                  // Target azimuth in sight
    env->observations[i++] = target_el * INV_HALF_PI;         // Target elevation in sight
    env->observations[i++] = range_km;                        // Range: 0=close, 1=2km+

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;                   // -1 to 1
    env->observations[i++] = horizon_visible;                 // -1 to 1
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);  // Closure rate
    // OBS_SIZE = 10
}

// Scheme 4: REALISTIC_ENEMY_STATE (13 obs)
// REALISTIC_RANGE + enemy pitch/roll/heading
void compute_obs_realistic_enemy_state(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Range in km
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Horizon visible
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    // Enemy Euler angles (relative to horizon)
    float enemy_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float enemy_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                              1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));

    // Enemy heading relative to player (+1 = pointing at player, -1 = pointing away)
    float enemy_heading_rel = target_aspect;  // Already computed as dot(opp_fwd, to_player)

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(norm3(p->vel) * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = pitch * INV_HALF_PI;
    env->observations[i++] = roll * INV_PI;

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = range_km;

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;
    env->observations[i++] = horizon_visible;
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);

    // Enemy state (3 obs) - NEW
    env->observations[i++] = enemy_pitch * INV_HALF_PI;  // Enemy nose angle vs horizon
    env->observations[i++] = enemy_roll * INV_PI;             // Enemy bank angle vs horizon
    env->observations[i++] = enemy_heading_rel;           // Pointing toward/away
    // OBS_SIZE = 13
}

// Scheme 5: REALISTIC_FULL (15 obs)
// REALISTIC_ENEMY_STATE + turn rate + G-loading
void compute_obs_realistic_full(Dogfight *env) {
    Plane *p = &env->player;
    Plane *o = &env->opponent;

    Quat q_inv = {p->ori.w, -p->ori.x, -p->ori.y, -p->ori.z};

    // Player Euler angles
    float pitch = asinf(clampf(2.0f * (p->ori.w * p->ori.y - p->ori.z * p->ori.x), -1.0f, 1.0f));
    float roll = atan2f(2.0f * (p->ori.w * p->ori.x + p->ori.y * p->ori.z),
                        1.0f - 2.0f * (p->ori.x * p->ori.x + p->ori.y * p->ori.y));

    // Target in body frame for gunsight
    Vec3 rel_pos = sub3(o->pos, p->pos);
    Vec3 rel_pos_body = quat_rotate(q_inv, rel_pos);
    float dist = norm3(rel_pos);

    float target_az = atan2f(rel_pos_body.y, rel_pos_body.x);
    float r_horiz = sqrtf(rel_pos_body.x * rel_pos_body.x + rel_pos_body.y * rel_pos_body.y);
    float target_el = atan2f(rel_pos_body.z, fmaxf(r_horiz, 1e-6f));

    // Range in km
    float range_km = clampf(dist / 2000.0f, 0.0f, 1.0f);

    // Opponent aspect
    Vec3 opp_fwd = quat_rotate(o->ori, vec3(1, 0, 0));
    Vec3 to_player = normalize3(sub3(p->pos, o->pos));
    float target_aspect = dot3(opp_fwd, to_player);

    // Horizon visible
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float horizon_visible = up.z;

    // Closure rate
    Vec3 rel_vel = sub3(p->vel, o->vel);
    float closure_rate = dot3(rel_vel, normalize3(rel_pos));

    // Enemy Euler angles
    float enemy_pitch = asinf(clampf(2.0f * (o->ori.w * o->ori.y - o->ori.z * o->ori.x), -1.0f, 1.0f));
    float enemy_roll = atan2f(2.0f * (o->ori.w * o->ori.x + o->ori.y * o->ori.z),
                              1.0f - 2.0f * (o->ori.x * o->ori.x + o->ori.y * o->ori.y));
    float enemy_heading_rel = target_aspect;

    // Turn rate from velocity change
    float speed = norm3(p->vel);
    float turn_rate_actual = 0.0f;
    if (speed > 10.0f) {
        Vec3 accel = mul3(sub3(p->vel, p->prev_vel), 1.0f / DT);
        Vec3 vel_dir = mul3(p->vel, 1.0f / speed);
        float accel_forward = dot3(accel, vel_dir);
        Vec3 accel_centripetal = sub3(accel, mul3(vel_dir, accel_forward));
        float centripetal_mag = norm3(accel_centripetal);
        turn_rate_actual = centripetal_mag / speed;  // ω = a/v
    }
    // Normalize turn rate: max ~0.5 rad/s (29°/s) for sustained turn
    float turn_rate_norm = clampf(turn_rate_actual / 0.5f, -1.0f, 1.0f);

    // G-loading: use physics-accurate p->g_force (aerodynamic forces)
    // Range: -1.5 to +6.0 G, normalize so 1G = 0, 6G = 1, -1.5G = -0.5
    float g_loading_norm = clampf((p->g_force - 1.0f) / 5.0f, -0.5f, 1.0f);

    int i = 0;
    // Instruments (4 obs)
    env->observations[i++] = clampf(speed * INV_MAX_SPEED, 0.0f, 1.0f);
    env->observations[i++] = p->pos.z * INV_WORLD_MAX_Z;
    env->observations[i++] = pitch * INV_HALF_PI;
    env->observations[i++] = roll * INV_PI;

    // Gunsight (3 obs)
    env->observations[i++] = target_az * INV_PI;
    env->observations[i++] = target_el * INV_HALF_PI;
    env->observations[i++] = range_km;

    // Visual cues (3 obs)
    env->observations[i++] = target_aspect;
    env->observations[i++] = horizon_visible;
    env->observations[i++] = clampf(closure_rate * INV_MAX_SPEED, -1.0f, 1.0f);

    // Enemy state (3 obs)
    env->observations[i++] = enemy_pitch * INV_HALF_PI;
    env->observations[i++] = enemy_roll * INV_PI;
    env->observations[i++] = enemy_heading_rel;

    // Own state (2 obs) - NEW
    env->observations[i++] = turn_rate_norm;    // How fast am I turning?
    env->observations[i++] = g_loading_norm;    // How hard am I pulling?
    // OBS_SIZE = 15
}

// Dispatcher function
void compute_observations(Dogfight *env) {
    switch (env->obs_scheme) {
        case OBS_ANGLES:               compute_obs_angles(env); break;
        case OBS_PURSUIT:              compute_obs_pursuit(env); break;
        case OBS_REALISTIC:            compute_obs_realistic(env); break;
        case OBS_REALISTIC_RANGE:      compute_obs_realistic_range(env); break;
        case OBS_REALISTIC_ENEMY_STATE: compute_obs_realistic_enemy_state(env); break;
        case OBS_REALISTIC_FULL:       compute_obs_realistic_full(env); break;
        default:                       compute_obs_angles(env); break;
    }
}

// ============================================================================
// Curriculum Learning: Stage-specific spawn functions
// ============================================================================

// Get current curriculum stage - now performance-based (df10)
// Stage advancement/demotion handled in add_log() based on recent kill rate
CurriculumStage get_curriculum_stage(Dogfight *env) {
    if (!env->curriculum_enabled) return CURRICULUM_FULL_RANDOM;
    if (env->curriculum_randomize) {
        // Random stage for eval mode - tests all difficulties
        return (CurriculumStage)(rand() % CURRICULUM_COUNT);
    }
    // Stage is managed by add_log() based on performance
    return env->stage;
}

// Stage 0: TAIL_CHASE - Opponent ahead, same heading (easiest)
void spawn_tail_chase(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead with offset giving ~15-25% chance of aligned spawn
    // At 300m, 5° gun cone = ~26m radius for hits
    // ±40/±30 gives avg offset ~25m = borderline hits, provides learning signal
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-40, 40),
        player_pos.z + rndf(-30, 30)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_STRAIGHT;
}

// Stage 1: HEAD_ON - Opponent coming toward us
void spawn_head_on(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
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

// Stage 6: CROSSING - 45 degree deflection shots (reduced from 90° - see CURRICULUM_PLANS.md)
// 90° deflection is historically nearly impossible; 45° is achievable with proper lead
void spawn_crossing(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
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

// Stage 3: VERTICAL - Above or below player
void spawn_vertical(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead, 200-400m above OR below
    float vert = rndf(0, 1) > 0.5f ? 1.0f : -1.0f;
    float alt_offset = vert * rndf(200, 400);
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-50, 50),
        clampf(player_pos.z + alt_offset, 300, 2500)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = AP_LEVEL;  // Maintain altitude
}

// Stage 4: MANEUVERING - Opponent does gentle turns (30°)
void spawn_maneuvering(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Random spawn position (similar to original)
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 500),
        player_pos.y + rndf(-100, 100),
        player_pos.z + rndf(-50, 50)
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    // Randomly choose turn direction - gentle 30° bank
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = AP_STAGE4_BANK_DEG * (M_PI / 180.0f);  // 30°
}

// Stage 5: FULL_RANDOM - Medium difficulty (360° spawn + random heading, 45° turns)
void spawn_full_random(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Random direction in 3D sphere (300-600m from player)
    float dist = rndf(300, 600);
    float theta = rndf(0, 2.0f * M_PI);      // Azimuth: 0-360°
    float phi = rndf(-0.3f, 0.3f);           // Elevation: ±17° (keep near level)

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 2500)
    );

    // Random velocity direction (not necessarily toward/away from player)
    float vel_theta = rndf(0, 2.0f * M_PI);
    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(vel_theta), speed * sinf(vel_theta), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);

    // Set orientation to match velocity direction (yaw rotation around Z)
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), vel_theta);

    // Use autopilot randomization (if configured)
    if (env->opponent_ap.randomize_on_reset) {
        autopilot_randomize(&env->opponent_ap);
    } else {
        // Default: uniform random mode with 45° turns
        float r = rndf(0, 1);
        if (r < 0.2f) env->opponent_ap.mode = AP_STRAIGHT;
        else if (r < 0.4f) env->opponent_ap.mode = AP_LEVEL;
        else if (r < 0.6f) env->opponent_ap.mode = AP_TURN_LEFT;
        else if (r < 0.8f) env->opponent_ap.mode = AP_TURN_RIGHT;
        else env->opponent_ap.mode = AP_CLIMB;
    }
    // Set 45° bank for stage 5 turns
    env->opponent_ap.target_bank = AP_STAGE5_BANK_DEG * (M_PI / 180.0f);
}

// Stage 6: HARD_MANEUVERING - Hard turns and weave patterns
void spawn_hard_maneuvering(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
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

// Stage 7: EVASIVE - Opponent reacts to player position
void spawn_evasive(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Spawn in various positions (like FULL_RANDOM)
    float dist = rndf(300, 500);
    float theta = rndf(0, 2.0f * M_PI);
    float phi = rndf(-0.3f, 0.3f);

    Vec3 opp_pos = vec3(
        player_pos.x + dist * cosf(theta) * cosf(phi),
        player_pos.y + dist * sinf(theta) * cosf(phi),
        clampf(player_pos.z + dist * sinf(phi), 300, 2500)
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
        env->opponent_ap.target_bank = AP_STAGE6_BANK_DEG * (M_PI / 180.0f);  // 60°
    }
}

// Master spawn function: dispatches to stage-specific spawner
void spawn_by_curriculum(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
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

    switch (env->stage) {
        case CURRICULUM_TAIL_CHASE:       spawn_tail_chase(env, player_pos, player_vel); break;
        case CURRICULUM_HEAD_ON:          spawn_head_on(env, player_pos, player_vel); break;
        case CURRICULUM_CROSSING:         spawn_crossing(env, player_pos, player_vel); break;
        case CURRICULUM_VERTICAL:         spawn_vertical(env, player_pos, player_vel); break;
        case CURRICULUM_MANEUVERING:      spawn_maneuvering(env, player_pos, player_vel); break;
        case CURRICULUM_FULL_RANDOM:      spawn_full_random(env, player_pos, player_vel); break;
        case CURRICULUM_HARD_MANEUVERING: spawn_hard_maneuvering(env, player_pos, player_vel); break;
        case CURRICULUM_EVASIVE:
        default:                          spawn_evasive(env, player_pos, player_vel); break;
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

void c_reset(Dogfight *env) {
    // Increment total episodes BEFORE determining stage (so first episode is 0)
    env->total_episodes++;

    env->tick = 0;
    env->episode_return = 0.0f;

    // Clear episode tracking
    env->kill = 0;
    env->episode_shots_fired = 0.0f;
    env->total_aileron_usage = 0.0f;
    env->aileron_bias = 0.0f;

    // Reset reward accumulators
    env->sum_r_closing = 0.0f;
    env->sum_r_speed = 0.0f;
    env->sum_r_neg_g = 0.0f;
    env->sum_r_rudder = 0.0f;
    env->sum_r_aim = 0.0f;
    env->death_reason = DEATH_NONE;

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

    // Gun cone for hit detection - stays fixed at 5°
    env->cos_gun_cone = cosf(env->gun_cone_angle);
    env->cos_gun_cone_2x = cosf(env->gun_cone_angle * 2.0f);

    // Anneal reward cone: start large (easy), shrink to gun cone (hard)
    float anneal_frac = fminf((float)env->total_episodes / (float)env->aim_anneal_episodes, 1.0f);
    env->reward_cone_angle = env->aim_cone_start + anneal_frac * (env->aim_cone_end - env->aim_cone_start);
    env->cos_reward_cone = cosf(env->reward_cone_angle);
    env->cos_reward_cone_2x = cosf(env->reward_cone_angle * 2.0f);

    // Spawn player at random position
    Vec3 pos = vec3(rndf(-500, 500), rndf(-500, 500), rndf(500, 1500));
    Vec3 vel = vec3(80, 0, 0);
    reset_plane(&env->player, pos, vel);

    // Spawn opponent based on curriculum stage (or legacy if disabled)
    if (env->curriculum_enabled) {
        spawn_by_curriculum(env, pos, vel);
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

// Respawn opponent at random position ahead of player
void respawn_opponent(Dogfight *env) {
    Plane *p = &env->player;
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));

    // Spawn 300-600m ahead, with some lateral offset
    Vec3 opp_pos = vec3(
        p->pos.x + fwd.x * rndf(300, 600) + rndf(-100, 100),
        p->pos.y + fwd.y * rndf(300, 600) + rndf(-100, 100),
        clampf(p->pos.z + rndf(-100, 100), 200, 2500)
    );
    Vec3 vel = vec3(80, 0, 0);
    reset_plane(&env->opponent, opp_pos, vel);

    // Reset autopilot PID state on respawn
    env->opponent_ap.prev_vz = 0.0f;
    env->opponent_ap.prev_bank_error = 0.0f;

    if (DEBUG >= 10) printf("=== RESPAWN ===\n");
    if (DEBUG >= 10) printf("player_pos=(%.1f, %.1f, %.1f)\n", p->pos.x, p->pos.y, p->pos.z);
    if (DEBUG >= 10) printf("player_fwd=(%.2f, %.2f, %.2f)\n", fwd.x, fwd.y, fwd.z);
    if (DEBUG >= 10) printf("new_opponent_pos=(%.1f, %.1f, %.1f)\n", opp_pos.x, opp_pos.y, opp_pos.z);
    if (DEBUG >= 10) printf("opponent_vel=(%.1f, %.1f, %.1f) NOTE: always +X!\n", vel.x, vel.y, vel.z);
    if (DEBUG >= 10) printf("respawn_dist=%.1f m\n", norm3(sub3(opp_pos, p->pos)));
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

    // Player uses full physics with actions
    step_plane_with_physics(&env->player, env->actions, DT);

    // Opponent uses autopilot (if not AP_STRAIGHT, uses full physics)
    if (env->opponent_ap.mode != AP_STRAIGHT) {
        float opp_actions[5];
        env->opponent_ap.threat_pos = env->player.pos;  // For AP_EVASIVE mode
        autopilot_step(&env->opponent_ap, &env->opponent, opp_actions, DT);
        step_plane_with_physics(&env->opponent, opp_actions, DT);
    } else {
        step_plane(&env->opponent, DT);
    }

    // Track aileron usage for monitoring (no death penalty - see BISECTION.md)
    env->total_aileron_usage += fabsf(env->actions[2]);

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

    // === Combat (Phase 5) ===
    Plane *p = &env->player;
    Plane *o = &env->opponent;
    float reward = 0.0f;

    // Decrement fire cooldowns
    if (p->fire_cooldown > 0) p->fire_cooldown--;
    if (o->fire_cooldown > 0) o->fire_cooldown--;

    // Player fires: action[4] > 0.5 and cooldown ready
    if (DEBUG >= 10) printf("trigger=%.3f, cooldown=%d\n", env->actions[4], p->fire_cooldown);
    if (env->actions[4] > 0.5f && p->fire_cooldown == 0) {
        p->fire_cooldown = FIRE_COOLDOWN;
        env->episode_shots_fired += 1.0f;
        if (DEBUG >= 10) printf("=== FIRED! episode_shots_fired=%.0f ===\n", env->episode_shots_fired);

        // Check if hit = kill = SUCCESS = terminal
        if (check_hit(p, o, env->cos_gun_cone)) {
            if (DEBUG >= 10) printf("*** KILL! ***\n");
            env->kill = 1;
            env->death_reason = DEATH_KILL;
            env->rewards[0] = 1.0f;
            env->episode_return += 1.0f;
            env->terminals[0] = 1;
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

    // 1. Closing velocity: approaching = good
    Vec3 rel_vel = sub3(p->vel, o->vel);
    Vec3 rel_pos_norm = normalize3(rel_pos);
    float closing_rate = dot3(rel_vel, rel_pos_norm);
    float r_closing = clampf(closing_rate * env->rcfg.closing_scale, -0.05f, 0.05f);
    reward += r_closing;

    // 2. Aim quality: continuous feedback for gun alignment
    Vec3 player_fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float aim_dot = dot3(rel_pos_norm, player_fwd);  // -1 to +1
    float aim_angle_deg = acosf(clampf(aim_dot, -1.0f, 1.0f)) * RAD_TO_DEG;
    float r_aim = 0.0f;
    if (dist < GUN_RANGE * 2.0f) {  // Only in engagement envelope (~1000m)
        float aim_quality = (aim_dot + 1.0f) * 0.5f;  // Remap [-1,1] to [0,1]
        r_aim = aim_quality * env->rcfg.aim_scale;
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
        r_stall = -(env->rcfg.speed_min - speed) * env->rcfg.stall;
    }
    reward += r_stall;

    // 5. Rudder penalty: prevent knife-edge climbing (small)
    float r_rudder = -fabsf(env->actions[3]) * env->rcfg.rudder;
    reward += r_rudder;

#if DEBUG >= 2
    // Track aiming diagnostics
    {
        float aim_angle_rad = acosf(clampf(aim_dot, -1.0f, 1.0f));
        if (aim_angle_rad < env->best_aim_angle) env->best_aim_angle = aim_angle_rad;
        if (aim_dot > env->cos_gun_cone) env->ticks_in_cone++;
        if (dist < env->closest_dist) env->closest_dist = dist;
    }
#endif

    // Accumulate for episode summary
    env->sum_r_closing += r_closing;
    env->sum_r_aim += r_aim;
    env->sum_r_neg_g += r_neg_g;
    env->sum_r_speed += r_stall;
    env->sum_r_rudder += r_rudder;

    if (DEBUG >= 4 && env->env_num == 0) printf("=== REWARD (df11) ===\n");
    if (DEBUG >= 4 && env->env_num == 0) printf("r_closing=%.4f (rate=%.1f m/s)\n", r_closing, closing_rate);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_aim=%.4f (aim_angle=%.1f deg, dist=%.1f)\n", r_aim, aim_angle_deg, dist);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_neg_g=%.5f (g=%.2f)\n", r_neg_g, p->g_force);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_stall=%.4f (speed=%.1f)\n", r_stall, speed);
    if (DEBUG >= 4 && env->env_num == 0) printf("r_rudder=%.5f (rud=%.2f)\n", r_rudder, env->actions[3]);
    if (DEBUG >= 4 && env->env_num == 0) printf("reward_total=%.4f\n", reward);

    if (DEBUG >= 10) printf("=== COMBAT ===\n");
    if (DEBUG >= 10) printf("aim_angle=%.1f deg (cone=5 deg)\n", aim_angle_deg);
    if (DEBUG >= 10) printf("dist_to_target=%.1f m (gun_range=500)\n", dist);
    if (DEBUG >= 10) printf("in_cone=%d, in_range=%d\n", aim_dot > env->cos_gun_cone, dist < GUN_RANGE);

    // Global reward clamping to prevent gradient explosion (restored for df8)
    reward = fmaxf(-1.0f, fminf(1.0f, reward));

    env->rewards[0] = reward;
    env->episode_return += reward;

    // Check bounds (player only)
    bool oob = fabsf(p->pos.x) > WORLD_HALF_X ||
               fabsf(p->pos.y) > WORLD_HALF_Y ||
               p->pos.z < 0 || p->pos.z > WORLD_MAX_Z;

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
        } else if (oob) {
            env->death_reason = DEATH_OOB;
        } else {
            env->death_reason = DEATH_TIMEOUT;
        }
        env->rewards[0] = (supersonic || p->pos.z <= 0) ? -1.0f : 0.0f;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
}

// Forward declaration for c_close (used in c_render)
void c_close(Dogfight *env);

// Draw airplane shape using lines - shows roll/pitch/yaw clearly
// Body frame: X=forward, Y=right, Z=up
void draw_plane_shape(Vec3 pos, Quat ori, Color body_color, Color wing_color) {
    // Body frame points (scaled for visibility: ~20m wingspan, ~25m length)
    Vec3 nose = vec3(15, 0, 0);
    Vec3 tail = vec3(-10, 0, 0);
    Vec3 left_wing = vec3(0, -12, 0);
    Vec3 right_wing = vec3(0, 12, 0);
    Vec3 vtail_top = vec3(-8, 0, 8);       // Vertical stabilizer
    Vec3 htail_left = vec3(-10, -5, 0);    // Horizontal stabilizer
    Vec3 htail_right = vec3(-10, 5, 0);

    // Rotate all points by orientation and translate to world position
    Vec3 nose_w = add3(pos, quat_rotate(ori, nose));
    Vec3 tail_w = add3(pos, quat_rotate(ori, tail));
    Vec3 lwing_w = add3(pos, quat_rotate(ori, left_wing));
    Vec3 rwing_w = add3(pos, quat_rotate(ori, right_wing));
    Vec3 vtop_w = add3(pos, quat_rotate(ori, vtail_top));
    Vec3 htl_w = add3(pos, quat_rotate(ori, htail_left));
    Vec3 htr_w = add3(pos, quat_rotate(ori, htail_right));

    // Convert to Raylib Vector3
    Vector3 nose_r = {nose_w.x, nose_w.y, nose_w.z};
    Vector3 tail_r = {tail_w.x, tail_w.y, tail_w.z};
    Vector3 lwing_r = {lwing_w.x, lwing_w.y, lwing_w.z};
    Vector3 rwing_r = {rwing_w.x, rwing_w.y, rwing_w.z};
    Vector3 vtop_r = {vtop_w.x, vtop_w.y, vtop_w.z};
    Vector3 htl_r = {htl_w.x, htl_w.y, htl_w.z};
    Vector3 htr_r = {htr_w.x, htr_w.y, htr_w.z};

    // Fuselage (nose to tail)
    DrawLine3D(nose_r, tail_r, body_color);

    // Main wings (left to right, through center for visibility)
    DrawLine3D(lwing_r, rwing_r, wing_color);
    // Wing to fuselage connections (makes it look more solid)
    DrawLine3D(lwing_r, nose_r, wing_color);
    DrawLine3D(rwing_r, nose_r, wing_color);

    // Vertical stabilizer (tail to top)
    DrawLine3D(tail_r, vtop_r, body_color);

    // Horizontal stabilizer
    DrawLine3D(htl_r, htr_r, body_color);
    DrawLine3D(htl_r, tail_r, body_color);
    DrawLine3D(htr_r, tail_r, body_color);

    // Small sphere at nose to show front clearly
    DrawSphere(nose_r, 2.0f, body_color);
}

void handle_camera_controls(Client *c) {
    Vector2 mouse = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        c->is_dragging = true;
        c->last_mouse_x = mouse.x;
        c->last_mouse_y = mouse.y;
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        c->is_dragging = false;
    }

    if (c->is_dragging) {
        float sensitivity = 0.005f;
        c->cam_azimuth -= (mouse.x - c->last_mouse_x) * sensitivity;
        c->cam_elevation += (mouse.y - c->last_mouse_y) * sensitivity;
        c->cam_elevation = clampf(c->cam_elevation, -1.4f, 1.4f);  // prevent gimbal lock
        c->last_mouse_x = mouse.x;
        c->last_mouse_y = mouse.y;
    }

    // Mouse wheel zoom
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        c->cam_distance = clampf(c->cam_distance - wheel * 10.0f, 30.0f, 300.0f);
    }
}

// Draw a single observation bar
// x, y: top-left position
// label: observation name
// value: the observation value
// is_01_range: true for [0,1] range, false for [-1,1] range
void draw_obs_bar(int x, int y, const char* label, float value, bool is_01_range) {
    // Draw label (fixed width)
    DrawText(label, x, y, 14, WHITE);

    // Bar dimensions
    int bar_x = x + 80;
    int bar_w = 150;
    int bar_h = 14;

    // Draw background
    DrawRectangle(bar_x, y, bar_w, bar_h, DARKGRAY);

    // Calculate fill position
    float norm_val;
    int fill_x, fill_w;

    if (is_01_range) {
        // [0, 1] range - fill from left
        norm_val = clampf(value, 0.0f, 1.0f);
        fill_x = bar_x;
        fill_w = (int)(norm_val * bar_w);
    } else {
        // [-1, 1] range - fill from center
        norm_val = clampf(value, -1.0f, 1.0f);
        int center = bar_x + bar_w / 2;
        if (norm_val >= 0) {
            fill_x = center;
            fill_w = (int)(norm_val * bar_w / 2);
        } else {
            fill_w = (int)(-norm_val * bar_w / 2);
            fill_x = center - fill_w;
        }
    }

    // Color based on magnitude
    Color fill_color = GREEN;
    if (fabsf(value) > 0.9f) fill_color = YELLOW;
    if (fabsf(value) > 1.0f) fill_color = RED;

    DrawRectangle(fill_x, y, fill_w, bar_h, fill_color);

    // Draw center line for [-1,1] range
    if (!is_01_range) {
        int center = bar_x + bar_w / 2;
        DrawLine(center, y, center, y + bar_h, WHITE);
    }

    // Draw value text
    DrawText(TextFormat("%+.2f", value), bar_x + bar_w + 5, y, 14, WHITE);
}

// Draw observation monitor showing all observation values as bars
void draw_obs_monitor(Dogfight *env) {
    int start_x = 900;
    int start_y = 10;
    int row_height = 18;

    const char** labels = NULL;
    int num_obs = env->obs_size;

    // Select labels based on scheme
    switch (env->obs_scheme) {
        case OBS_ANGLES:
            labels = OBS_LABELS_ANGLES;
            break;
        case OBS_PURSUIT:
            labels = OBS_LABELS_PURSUIT;
            break;
        case OBS_REALISTIC:
            labels = OBS_LABELS_REALISTIC;
            break;
        case OBS_REALISTIC_RANGE:
            labels = OBS_LABELS_REALISTIC_RANGE;
            break;
        case OBS_REALISTIC_ENEMY_STATE:
            labels = OBS_LABELS_REALISTIC_ENEMY_STATE;
            break;
        case OBS_REALISTIC_FULL:
            labels = OBS_LABELS_REALISTIC_FULL;
            break;
        default:
            labels = OBS_LABELS_ANGLES;
            break;
    }

    // Title
    DrawText(TextFormat("OBS (scheme %d)", env->obs_scheme),
             start_x, start_y, 16, YELLOW);
    start_y += 22;

    // Draw each observation bar
    for (int i = 0; i < num_obs; i++) {
        float val = env->observations[i];
        // Determine if this observation is [0,1] range
        // Based on observation scheme and index:
        // - Scheme 0 (ANGLES): index 3 (speed) is [0,1]
        // - Scheme 1 (PURSUIT): indices 0 (speed), 1 (potential), 4 (energy) are [0,1]
        // - Scheme 2-5 (REALISTIC*): indices 0 (airspeed), 1 (altitude) are [0,1]
        bool is_01 = false;
        switch (env->obs_scheme) {
            case OBS_ANGLES:
                is_01 = (i == 3);  // speed
                break;
            case OBS_PURSUIT:
                is_01 = (i == 0 || i == 1 || i == 4);  // speed, potential, energy
                break;
            case OBS_REALISTIC:
            case OBS_REALISTIC_RANGE:
            case OBS_REALISTIC_ENEMY_STATE:
            case OBS_REALISTIC_FULL:
                is_01 = (i == 0 || i == 1);  // airspeed, altitude
                // Also range_km (index 6) is [0,1]
                if (env->obs_scheme != OBS_REALISTIC && i == 6) is_01 = true;
                break;
            default:
                break;
        }
        int y = start_y + i * row_height;
        draw_obs_bar(start_x, y, labels[i], val, is_01);

        // Draw red arrow for highlighted observations
        if (env->obs_highlight[i]) {
            // Draw arrow pointing right at the label (triangle)
            int arrow_x = start_x - 20;
            int arrow_y = y + 7;  // Center vertically
            // Triangle pointing right: 3 points
            DrawTriangle(
                (Vector2){arrow_x, arrow_y - 5},      // Top
                (Vector2){arrow_x, arrow_y + 5},      // Bottom
                (Vector2){arrow_x + 12, arrow_y},     // Tip (right)
                RED
            );
        }
    }
}

void c_render(Dogfight *env) {
    // 1. Lazy initialization
    if (env->client == NULL) {
        env->client = (Client *)calloc(1, sizeof(Client));
        env->client->width = 1280;
        env->client->height = 720;
        env->client->cam_distance = 80.0f;
        env->client->cam_azimuth = 0.0f;
        env->client->cam_elevation = 0.3f;
        env->client->is_dragging = false;

        InitWindow(1280, 720, "Dogfight");
        SetTargetFPS(60);

        // Z-up coordinate system
        env->client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
        env->client->camera.fovy = 45.0f;
        env->client->camera.projection = CAMERA_PERSPECTIVE;
    }

    // 2. Handle window close
    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    // 3. Handle mouse controls for camera orbit
    handle_camera_controls(env->client);

    // 4. Update chase camera
    Plane *p = &env->player;
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    float dist = env->client->cam_distance;

    // Apply orbit offsets from mouse drag
    float az = env->client->cam_azimuth;
    float el = env->client->cam_elevation;

    // Base chase position (behind and above player)
    float cam_x = p->pos.x - fwd.x * dist * cosf(el) * cosf(az) + fwd.y * dist * sinf(az);
    float cam_y = p->pos.y - fwd.y * dist * cosf(el) * cosf(az) - fwd.x * dist * sinf(az);
    float cam_z = p->pos.z + dist * sinf(el) + 20.0f;

    env->client->camera.position = (Vector3){cam_x, cam_y, cam_z};
    env->client->camera.target = (Vector3){p->pos.x, p->pos.y, p->pos.z};

    // 5. Begin drawing
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});  // Dark blue-green sky

    // Set clip planes for long-range visibility (default far=1000 is too close)
    rlSetClipPlanes(1.0, 10000.0);  // near=1m, far=10km
    BeginMode3D(env->client->camera);

    // 6. Draw ground plane at z=0 (XY plane, since we use Z-up)
    // DrawPlane uses raylib's Y-up convention (XZ plane), so we draw triangles instead
    Vector3 g1 = {-2000, -2000, 0};
    Vector3 g2 = {2000, -2000, 0};
    Vector3 g3 = {2000, 2000, 0};
    Vector3 g4 = {-2000, 2000, 0};
    Color ground_color = (Color){20, 60, 20, 255};
    DrawTriangle3D(g1, g2, g3, ground_color);
    DrawTriangle3D(g1, g3, g4, ground_color);

    // 7. Draw world bounds wireframe
    // Bounds: X +/-2000, Y +/-2000, Z 0-3000 -> center at (0, 0, 1500)
    DrawCubeWires((Vector3){0, 0, 1500}, 4000, 4000, 3000, (Color){100, 100, 100, 255});

    // 8. Draw player plane (cyan wireframe airplane)
    Color cyan = {0, 255, 255, 255};
    Color light_cyan = {100, 255, 255, 255};
    draw_plane_shape(p->pos, p->ori, cyan, light_cyan);

    // 9. Draw opponent plane (red wireframe airplane)
    Plane *o = &env->opponent;
    draw_plane_shape(o->pos, o->ori, RED, ORANGE);

    // 10. Draw tracer when firing (cooldown just set = just fired)
    if (p->fire_cooldown >= FIRE_COOLDOWN - 2) {  // Show for 2 frames
        Vec3 nose = add3(p->pos, quat_rotate(p->ori, vec3(15, 0, 0)));
        Vec3 tracer_end = add3(p->pos, quat_rotate(p->ori, vec3(GUN_RANGE, 0, 0)));
        Vector3 nose_r = {nose.x, nose.y, nose.z};
        Vector3 end_r = {tracer_end.x, tracer_end.y, tracer_end.z};
        DrawLine3D(nose_r, end_r, YELLOW);
    }

    EndMode3D();

    // 10. Draw HUD
    float speed = norm3(p->vel);
    float dist_to_opp = norm3(sub3(o->pos, p->pos));

    DrawText(TextFormat("Speed: %.0f m/s", speed), 10, 10, 20, WHITE);
    DrawText(TextFormat("Altitude: %.0f m", p->pos.z), 10, 40, 20, WHITE);
    DrawText(TextFormat("Throttle: %.0f%%", p->throttle * 100.0f), 10, 70, 20, WHITE);
    DrawText(TextFormat("Distance: %.0f m", dist_to_opp), 10, 100, 20, WHITE);
    DrawText(TextFormat("Tick: %d / %d", env->tick, env->max_steps), 10, 130, 20, WHITE);
    DrawText(TextFormat("Return: %.2f", env->episode_return), 10, 160, 20, WHITE);
    DrawText(TextFormat("Perf: %.1f%% | Shots: %.0f", env->log.perf / fmaxf(env->log.n, 1.0f) * 100.0f, env->log.shots_fired), 10, 190, 20, YELLOW);

    // 11. Draw observation monitor (right side)
    draw_obs_monitor(env);

    // Controls hint
    DrawText("Mouse drag: Orbit | Scroll: Zoom | ESC: Exit", 10, (int)env->client->height - 30, 16, GRAY);

    EndDrawing();
}

void c_close(Dogfight *env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

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
    int tick           // = 0, environment tick
) {
    // Player state
    env->player.pos = vec3(p_px, p_py, p_pz);
    env->player.vel = vec3(p_vx, p_vy, p_vz);
    env->player.ori = quat(p_ow, p_ox, p_oy, p_oz);
    quat_normalize(&env->player.ori);
    env->player.throttle = p_throttle;
    env->player.fire_cooldown = 0;
    env->player.yaw_from_rudder = 0.0f;  // Reset accumulated rudder yaw

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
    env->opponent.fire_cooldown = 0;
    env->opponent.yaw_from_rudder = 0.0f;  // Reset accumulated rudder yaw

    // Environment state
    env->tick = tick;
    env->episode_return = 0.0f;

    compute_observations(env);
}
