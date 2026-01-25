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

typedef enum {
    OBS_MOMENTUM = 0,           // BASELINE: body-frame vel + omega + AoA + energy (15 obs)
    OBS_MOMENTUM_BETA = 1,      // + sideslip angle (16 obs)
    OBS_MOMENTUM_GFORCE = 2,    // + G-force (16 obs)
    OBS_MOMENTUM_FULL = 3,      // + sideslip + G + throttle + tgt rates (19 obs)
    OBS_MINIMAL = 4,            // stripped down essentials (11 obs)
    OBS_CARTESIAN = 5,          // cartesian target position (15 obs)
    OBS_DRONE_STYLE = 6,        // + quaternion + up vector (22 obs)
    OBS_QBAR = 7,               // + dynamic pressure (16 obs)
    OBS_KITCHEN_SINK = 8,       // everything (25 obs)
    OBS_SCHEME_COUNT
} ObsScheme;

static const int OBS_SIZES[OBS_SCHEME_COUNT] = {15, 16, 16, 19, 11, 15, 22, 16, 25};

typedef enum {
    CURRICULUM_TAIL_CHASE = 0,       // Easiest: opponent ahead, same heading
    CURRICULUM_HEAD_ON,              // Opponent coming toward us
    CURRICULUM_VERTICAL,             // Above or below player
    CURRICULUM_MANEUVERING,          // Opponent does gentle 30° turns
    CURRICULUM_OFFSET_MANEUVERING,   // Large lateral/vertical offset, same heading
    CURRICULUM_ANGLED_MANEUVERING,   // Offset + different heading (±45°)
    CURRICULUM_FULL_RANDOM,          // 360° spawn, random heading, 45° turns
    CURRICULUM_HARD_MANEUVERING,     // 60° turns + weave patterns
    CURRICULUM_CROSSING,             // 45 degree deflection shots
    CURRICULUM_EVASIVE,              // Reactive evasion (hardest)
    CURRICULUM_COUNT
} CurriculumStage;

// Stage difficulty weights for composite metric (higher = harder = more valuable)
// Updated 2026-01-24 to include intermediate stages (see CLAUDE.md todo)
static const float STAGE_WEIGHTS[CURRICULUM_COUNT] = {
    0.20f,  // TAIL_CHASE - trivial
    0.30f,  // HEAD_ON - easy
    0.40f,  // VERTICAL - medium
    0.50f,  // MANEUVERING - gentle 30° turns
    0.52f,  // OFFSET_MANEUVERING - large position offsets
    0.58f,  // ANGLED_MANEUVERING - different heading (±45°)
    0.65f,  // FULL_RANDOM - 360° spawn, random heading, 45° turns
    0.80f,  // HARD_MANEUVERING - 60° turns + weave
    0.90f,  // CROSSING - 45° deflection shots
    1.00f   // EVASIVE - reactive opponent
};

#define DT 0.02f

#define WORLD_HALF_X 2000.0f
#define WORLD_HALF_Y 2000.0f
#define WORLD_MAX_Z 3000.0f
#define MAX_SPEED 250.0f

#define INV_WORLD_HALF_X 0.0005f       // 1/2000
#define INV_WORLD_HALF_Y 0.0005f       // 1/2000
#define INV_WORLD_MAX_Z  0.000333333f  // 1/3000
#define INV_MAX_SPEED    0.004f        // 1/250
#define INV_PI           0.31830988618f // 1/PI
#define INV_HALF_PI      0.63661977236f // 2/PI (i.e., 1/(PI*0.5))

#define GUN_RANGE 500.0f       // meters
#define INV_GUN_RANGE 0.002f   // 1/500
#define GUN_CONE_ANGLE 0.087f  // ~5 degrees in radians
#define FIRE_COOLDOWN 10       // ticks (0.2 seconds at 50Hz)

typedef struct Log {
    float episode_return;
    float episode_length;
    float score;           // 1.0 on kill, 0.0 on failure
    float perf;            // Raw kills (becomes kill_rate after vec_log divides by n)
    float shots_fired;
    float accuracy;
    float stage;

    // RAW SUMS - exported to Python, become correct averages after vec_log divides by n
    float total_stage_weight;       // Sum of stage weights (exported as avg_stage_weight)
    float total_abs_bias;           // Sum of |aileron_bias| (exported as avg_abs_bias)
    float stage_sum;                // Sum of stages (exported as avg_stage)

    // PER-ENV RATIOS - for C debugging only, NOT exported (garbage after vec_log aggregation)
    float avg_stage_weight;         // = total_stage_weight / n (per-env only)
    float avg_abs_bias;             // = total_abs_bias / n (per-env only)
    float avg_stage;                // = stage_sum / n (per-env only)
    float kill_rate;                // = perf / n (per-env only - Python uses 'perf' instead)
    float ultimate;                 // = kill_rate * avg_stage_weight (per-env only)
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
    // Thresholds
    float speed_min;         // Stall threshold (default 50.0)
} RewardConfig;

typedef struct Client {
    Camera3D camera;
    float width;
    float height;

    float cam_distance;
    float cam_azimuth;
    float cam_elevation;
    bool is_dragging;
    float last_mouse_x;
    float last_mouse_y;

    Model plane_model;
    Texture2D plane_texture;
    bool model_loaded;
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
    CurriculumStage stage;      // Current difficulty stage (set globally by Python)
    float curriculum_target;    // Float 0.0-7.0 for probabilistic stage assignment
    int is_initialized;         // Flag to preserve curriculum state across re-init (for Multiprocessing)
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
    // Debug
    int env_num;                // Environment index (for filtering debug output)
    // Observation highlighting (for visual debugging)
    unsigned char obs_highlight[25];  // 1 = highlight this observation with red arrow (max scheme is 25 obs)
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

    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));
}

void set_obs_highlight(Dogfight *env, int *indices, int count) {
    memset(env->obs_highlight, 0, sizeof(env->obs_highlight));
    for (int i = 0; i < count && i < 25; i++) {
        if (indices[i] >= 0 && indices[i] < 25) {
            env->obs_highlight[indices[i]] = 1;
        }
    }
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
    env->log.score += env->rewards[0];
    env->log.shots_fired += env->episode_shots_fired;
    env->log.accuracy = (env->log.shots_fired > 0.0f) ? (env->log.perf / env->log.shots_fired * 100.0f) : 0.0f;
    env->log.stage = (float)env->stage;

    env->log.total_stage_weight += STAGE_WEIGHTS[env->stage]; // coeffs to scale metrics based on difficulty
    env->log.total_abs_bias += fabsf(env->aileron_bias);
    env->log.stage_sum += (float)env->stage;  // Accumulate for avg_stage
    env->log.n += 1.0f;
    env->log.kill_rate = env->log.perf / fmaxf(env->log.n, 1.0f);
    env->log.avg_stage = env->log.stage_sum / env->log.n;
    env->log.avg_abs_bias = env->log.total_abs_bias / env->log.n;
    env->log.avg_stage_weight = env->log.total_stage_weight / env->log.n;

    // Ultimate = kill_rate * difficulty (no bias penalty)
    env->log.ultimate = env->log.kill_rate * env->log.avg_stage_weight;

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
void spawn_tail_chase(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 200-400m ahead with offset giving ~10-20% chance of aligned spawn
    // At 300m, 5° gun cone = ~26m radius for hits
    // ±50/±38 gives avg offset ~31m = requires minor adjustment
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-50, 50),
        player_pos.z + rndf(-38, 38)
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

// Stage 8: CROSSING - 45 degree deflection shots (reduced from 90° - see CURRICULUM_PLANS.md)
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

// Stage 2: VERTICAL - Above or below player
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

// Stage 3: MANEUVERING - Opponent does gentle turns (30°)
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

// Stage 4: OFFSET_MANEUVERING - Large lateral/vertical offset, same heading
// Teaches: Finding and tracking targets not directly in front
void spawn_offset_maneuvering(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    // Opponent 150-300m ahead with LARGE lateral/vertical offset
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(150, 300),
        player_pos.y + rndf(-250, 250),   // Large lateral - can be way to the side
        clampf(player_pos.z + rndf(-200, 200), 300, 2500)  // Large vertical
    );
    reset_plane(&env->opponent, opp_pos, player_vel);
    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = AP_STAGE4_BANK_DEG * (M_PI / 180.0f);  // 30°
}

// Stage 5: ANGLED_MANEUVERING - Offset + different heading (±45°)
// Teaches: Pursuit geometry when target isn't flying your direction
void spawn_angled_maneuvering(Dogfight *env, Vec3 player_pos, Vec3 player_vel) {
    Vec3 opp_pos = vec3(
        player_pos.x + rndf(200, 400),
        player_pos.y + rndf(-200, 200),
        clampf(player_pos.z + rndf(-150, 150), 300, 2500)
    );

    // Heading offset: ±45° from player
    float heading_offset = rndf(-0.785f, 0.785f);  // ±45° in radians
    float player_heading = atan2f(player_vel.y, player_vel.x);
    float opp_heading = player_heading + heading_offset;

    float speed = norm3(player_vel);
    Vec3 opp_vel = vec3(speed * cosf(opp_heading), speed * sinf(opp_heading), 0);

    reset_plane(&env->opponent, opp_pos, opp_vel);
    env->opponent.ori = quat_from_axis_angle(vec3(0, 0, 1), opp_heading);

    env->opponent_ap.mode = rndf(0, 1) > 0.5f ? AP_TURN_LEFT : AP_TURN_RIGHT;
    env->opponent_ap.target_bank = AP_STAGE4_BANK_DEG * (M_PI / 180.0f);  // 30°
}

// Stage 6: FULL_RANDOM - Medium-hard (360° spawn + random heading, 45° turns)
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

// Stage 7: HARD_MANEUVERING - Hard turns and weave patterns
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

// Stage 9: EVASIVE - Opponent reacts to player position (hardest)
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
        case CURRICULUM_TAIL_CHASE:         spawn_tail_chase(env, player_pos, player_vel); break;
        case CURRICULUM_HEAD_ON:            spawn_head_on(env, player_pos, player_vel); break;
        case CURRICULUM_VERTICAL:           spawn_vertical(env, player_pos, player_vel); break;
        case CURRICULUM_MANEUVERING:        spawn_maneuvering(env, player_pos, player_vel); break;
        case CURRICULUM_OFFSET_MANEUVERING: spawn_offset_maneuvering(env, player_pos, player_vel); break;
        case CURRICULUM_ANGLED_MANEUVERING: spawn_angled_maneuvering(env, player_pos, player_vel); break;
        case CURRICULUM_FULL_RANDOM:        spawn_full_random(env, player_pos, player_vel); break;
        case CURRICULUM_HARD_MANEUVERING:   spawn_hard_maneuvering(env, player_pos, player_vel); break;
        case CURRICULUM_CROSSING:           spawn_crossing(env, player_pos, player_vel); break;
        case CURRICULUM_EVASIVE:
        default:                            spawn_evasive(env, player_pos, player_vel); break;
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

// Set curriculum target (float 0.0-7.0) for probabilistic stage assignment
void set_curriculum_target(Dogfight *env, float target) {
    env->curriculum_target = fminf(fmaxf(target, 0.0f), (float)(CURRICULUM_COUNT - 1));
}

// ============================================================================

void c_reset(Dogfight *env) {
    // Curriculum stage is now managed globally by Python based on aggregate kill_rate
    // (see set_curriculum_stage() called from training loop)

    env->total_episodes++;

    env->tick = 0;
    env->episode_return = 0.0f;

    // Clear episode tracking (safe to clear kill after curriculum used it)
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
        r_stall = -(env->rcfg.speed_min - speed) * PENALTY_STALL;
    }
    reward += r_stall;

    // 5. Rudder penalty: prevent knife-edge climbing (small)
    float r_rudder = -fabsf(env->actions[3]) * PENALTY_RUDDER;
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
        env->rewards[0] = (supersonic || p->pos.z <= 0 || env->tick >= env->max_steps) ? -1.0f : 0.0f;
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }

    compute_observations(env);
#if DEBUG >= 5
    print_observations(env);
#endif
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
    int tick           // = 0, environment tick
) {
    env->player.pos = vec3(p_px, p_py, p_pz);
    env->player.vel = vec3(p_vx, p_vy, p_vz);
    env->player.prev_vel = vec3(p_vx, p_vy, p_vz);  // Initialize to current (no accel)
    env->player.omega = vec3(0, 0, 0);  // No angular velocity
    env->player.ori = quat(p_ow, p_ox, p_oy, p_oz);
    quat_normalize(&env->player.ori);
    env->player.throttle = p_throttle;
    env->player.fire_cooldown = 0;
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
    env->opponent.fire_cooldown = 0;
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
