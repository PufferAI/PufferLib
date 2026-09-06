#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"
typedef float obs_t;
#define PUF_HAS_BOT_POLICY
#include "pufferenv.h"

#define NUM_ACTIONS 5
#define NUM_BULLETS 16
#define EGO_FEATURES 14
#define OTHER_FEATURES 8

static const float ACCEL_VALUES[4] = {
    -2.0f, -1.0f, 0, 1.0f
};

static const float TURN_VALUES[9] = {
    -10.0f, -6.0f, -3.0f, -1.0f, 0, 1.0f, 3.0f, 6.0f, 10.0f
};

static const float GUN_TURN_VALUES[11] = {
    -20.0f, -10.0f, -5.0f, -3.0f, -1.0f, 0, 1.0f, 3.0f, 5.0f, 15.0f, 20.0f
};

static const float RADAR_TURN_VALUES[11] = {
    -45.0f, -25.0f, -10.0f, -5.0f,  -1.0f, 0, 1.0f, 5.0f, 10.0f, 25.0f, 45.0f
};
static const float FIREPOWER_VALUES[6] = {
    0, 0.1f, 0.5f, 1.0f, 2.0f, 3.0f
};
float cos_deg(float deg) {
    return cosf(deg * 3.14159265358979323846f / 180.0f);
}

float sin_deg(float deg) {
    return sinf(deg * 3.14159265358979323846f / 180.0f);
}

typedef struct BotMem BotMem;  // defined in bots.h

typedef struct Log Log;
struct Log {
    float perf;             // bots killed this episode
    float episode_return;
    float episode_length;
    float score;            // damage dealt this episode
    float damage_received;  // starting energy - current energy at episode end
    float melee_damage_inflicted;
    float damage_taken;
    float range_damage_inflicted;
    // Per-slot scores for match() scoring + selfplay sanity-check. In selfplay
    // both should average to ~0.5; in match A=policy 0, B=policy 1, policy_0_score is
    // the win rate of policy A. Each game contributes 1.0 worth of credit total
    // (win=1.0 to winner, draw=0.5 each). We scale by num_agents on accumulation
    // so the eval_log mean (sum / n where n increments by num_agents per ep)
    // equals win_rate directly.
    float policy_0_score;
    float policy_1_score;
    float draw_rate;
    // Opponent action-noise curriculum faced this episode (pre-decay).
    float bot_cl_noise;
    float hist_cl_noise;
    // Win credit * (1 - noise). Full credit only vs annealed (noise=0) opps.
    float cl_perf;
    float n;
};

typedef struct Bullet Bullet;
struct Bullet {
    float x;
    float y;
    float heading;
    float firepower;
    bool live;
};

typedef struct Robot Robot;
struct Robot {
    float x;
    float y;
    float v;
    float heading;
    float gun_heading;
    float radar_heading_prev;
    float radar_heading;
    float speed_mult;
    float handling_mult;
    float power_mult;
    float reward_melee_damage_inflicted;
    float reward_damage_taken;
    float reward_range_damage_inflicted;
    int bullet_idx;
    float gun_heat;
    float energy;
};

typedef struct Client Client;
typedef Env Robocode;

#define OBS_SIZE (EGO_FEATURES + OTHER_FEATURES)
#define NUM_ATNS 5
#define ACT_SIZES {4, 9, 11, 11, 6}

struct Env {
    Client* client;
    Agent agents[2];

    int num_agents;
    int num_bots;
    int tick;
    int max_ticks;   // episode timeout; configured per-run via [env].max_ticks
    int width;
    int height;
    Robot* robots;
    Bullet* bullets;
    Log log;
    Log* logs;
    // reward_damage is kept as a legacy config field. The explicit shaped
    // damage coefficients are fixed per env config / sweep trial and copied
    // into each learning slot on reset so policies can condition on them.
    float reward_damage;
    float reward_spot;
    float reward_melee_damage_inflicted;
    float reward_damage_taken;
    float reward_range_damage_inflicted;
    float reward_melee_damage_inflicted_slot_0;
    float reward_damage_taken_slot_0;
    float reward_range_damage_inflicted_slot_0;
    float reward_melee_damage_inflicted_slot_1;
    float reward_damage_taken_slot_1;
    float reward_range_damage_inflicted_slot_1;
    float dr;
    int bot_policy;
    // Optional second-bot policy for bot-vs-bot harnesses (num_agents=0).
    // <0 means both bots use bot_policy (normal train/eval).
    int bot_policy_1;
    // Bot-vs-bot result when num_agents==0: -2 ongoing, 0/1 winner idx, -1 draw.
    int bot_match_winner;
    BotMem* bot_mems;        // per-bot scratch (allocated by bots.h)

    // Opponent action-noise curriculum. With prob bot_cl_noise, scripted bots
    // take uniform discrete actions; with prob hist_cl_noise, hist slot-1
    // actions are overwritten. Primary win decays the active noise by *_decay.
    float bot_cl_noise;
    float bot_cl_decay;
    float hist_cl_noise;
    float hist_cl_decay;

    // Selfplay-pool tagging. tag = 0 means pure selfplay (both slots = primary
    // policy). tag > 0 means historical: slot 0 = primary, slot 1 = frozen
    // historical opponent. boundary_reached is set on game-end so the trainer
    // can swap frozen banks only between games.
    int tag;
    int boundary_reached;

    unsigned int rng;
};

static inline void puf_set_bot_policy(Env* env, int bot_policy) {
    env->bot_policy = bot_policy;
}

void init(Robocode* env);

static inline float robocode_get_float(Dict* kwargs, const char* key, float default_value) {
    for (int i = 0; i < kwargs->size; i++) {
        if (strcmp(kwargs->items[i].key, key) == 0) {
            return (float)kwargs->items[i].value;
        }
    }
    return default_value;
}

void puf_init(Env* env, Dict* kwargs) {
    env->width = dict_get(kwargs, "width");
    env->height = dict_get(kwargs, "height");
    env->num_agents = dict_get(kwargs, "num_agents");
    env->num_bots = dict_get(kwargs, "num_bots");
    env->max_ticks = dict_get(kwargs, "max_ticks");
    env->reward_damage = robocode_get_float(kwargs, "reward_damage", 0.0f);
    env->reward_spot = robocode_get_float(kwargs, "reward_spot", 0.0f);
    env->reward_melee_damage_inflicted = robocode_get_float(kwargs, "reward_melee_damage_inflicted", 0.0f);
    env->reward_damage_taken = robocode_get_float(kwargs, "reward_damage_taken", 0.0f);
    env->reward_range_damage_inflicted = robocode_get_float(kwargs, "reward_range_damage_inflicted", 0.0f);
    env->reward_melee_damage_inflicted_slot_0 = robocode_get_float(kwargs,
        "reward_melee_damage_inflicted_slot_0", env->reward_melee_damage_inflicted);
    env->reward_damage_taken_slot_0 = robocode_get_float(kwargs,
        "reward_damage_taken_slot_0", env->reward_damage_taken);
    env->reward_range_damage_inflicted_slot_0 = robocode_get_float(kwargs,
        "reward_range_damage_inflicted_slot_0", env->reward_range_damage_inflicted);
    env->reward_melee_damage_inflicted_slot_1 = robocode_get_float(kwargs,
        "reward_melee_damage_inflicted_slot_1", env->reward_melee_damage_inflicted);
    env->reward_damage_taken_slot_1 = robocode_get_float(kwargs,
        "reward_damage_taken_slot_1", env->reward_damage_taken);
    env->reward_range_damage_inflicted_slot_1 = robocode_get_float(kwargs,
        "reward_range_damage_inflicted_slot_1", env->reward_range_damage_inflicted);
    env->dr = robocode_get_float(kwargs, "dr", 0.0f);
    env->bot_policy = dict_get(kwargs, "bot_policy");
    env->bot_policy_1 = (int)robocode_get_float(kwargs, "bot_policy_1", -1.0f);
    env->bot_match_winner = -2;
    env->bot_cl_noise = robocode_get_float(kwargs, "bot_cl_noise", 0.0f);
    env->bot_cl_decay = robocode_get_float(kwargs, "bot_cl_decay", 0.0f);
    env->hist_cl_noise = robocode_get_float(kwargs, "hist_cl_noise", 0.0f);
    env->hist_cl_decay = robocode_get_float(kwargs, "hist_cl_decay", 0.0f);
    if (env->bot_cl_noise < 0.0f) env->bot_cl_noise = 0.0f;
    if (env->bot_cl_noise > 1.0f) env->bot_cl_noise = 1.0f;
    if (env->hist_cl_noise < 0.0f) env->hist_cl_noise = 0.0f;
    if (env->hist_cl_noise > 1.0f) env->hist_cl_noise = 1.0f;
    env->agents[0].policy = 0;
    env->agents[1].policy = 1;
    env->agents[0].action_mask = NULL;
    env->agents[1].action_mask = NULL;
    init(env);
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "damage_received", log->damage_received);
    dict_set(out, "melee_damage_inflicted", log->melee_damage_inflicted);
    dict_set(out, "damage_taken", log->damage_taken);
    dict_set(out, "range_damage_inflicted", log->range_damage_inflicted);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "policy_0_score", log->policy_0_score);
    dict_set(out, "policy_1_score", log->policy_1_score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "bot_cl_noise", log->bot_cl_noise);
    dict_set(out, "hist_cl_noise", log->hist_cl_noise);
    dict_set(out, "cl_perf", log->cl_perf);
    dict_set(out, "n", log->n);
}

static inline void bot_mems_alloc(Robocode* env);
static inline void bot_mems_free(Robocode* env);
static inline void bot_mems_episode_reset(Robocode* env);

void init(Robocode* env){
    int total_robots = env->num_agents + env->num_bots;
    env->robots = (Robot*)calloc(total_robots, sizeof(Robot));
    env->bullets = (Bullet*)calloc(NUM_BULLETS*total_robots, sizeof(Bullet));
    env->logs = (Log*)calloc(env->num_agents, sizeof(Log));
    bot_mems_alloc(env);
}

void puf_close(Robocode* env) {
    free(env->robots);
    free(env->bullets);
    free(env->logs);
    bot_mems_free(env);
}

void add_log(Robocode* env) {
    // Called at episode end. Finalize damage_received from current energy,
    // then fold per-agent running logs into the aggregate env->log.
    for (int i = 0; i < env->num_agents; i++) {
        env->logs[i].damage_received = 100.0f - (float)env->robots[i].energy;
        env->log.perf            += env->logs[i].perf;
        env->log.episode_return  += env->logs[i].episode_return;
        env->log.episode_length  += env->logs[i].episode_length;
        env->log.score                   += env->logs[i].score;
        env->log.damage_received         += env->logs[i].damage_received;
        env->log.melee_damage_inflicted  += env->logs[i].melee_damage_inflicted;
        env->log.damage_taken            += env->logs[i].damage_taken;
        env->log.range_damage_inflicted  += env->logs[i].range_damage_inflicted;
        env->log.bot_cl_noise            += env->logs[i].bot_cl_noise;
        env->log.hist_cl_noise           += env->logs[i].hist_cl_noise;
        env->log.n                       += 1.0f;
    }
}

bool segment_intersects_aabb(
    float x0, float y0,
    float x1, float y1,
    float left, float right,
    float bottom, float top
) {
    float tmin = 0.0f;
    float tmax = 1.0f;

    float dx = x1 - x0;
    float dy = y1 - y0;

    if (fabsf(dx) < 1e-8f) {
        if (x0 < left || x0 > right) return false;
    } else {
        float tx1 = (left  - x0) / dx;
        float tx2 = (right - x0) / dx;
        if (tx1 > tx2) { float tmp = tx1; tx1 = tx2; tx2 = tmp; }
        if (tx1 > tmin) tmin = tx1;
        if (tx2 < tmax) tmax = tx2;
        if (tmin > tmax) return false;
    }

    if (fabsf(dy) < 1e-8f) {
        if (y0 < bottom || y0 > top) return false;
    } else {
        float ty1 = (bottom - y0) / dy;
        float ty2 = (top    - y0) / dy;
        if (ty1 > ty2) { float tmp = ty1; ty1 = ty2; ty2 = tmp; }
        if (ty1 > tmin) tmin = ty1;
        if (ty2 < tmax) tmax = ty2;
        if (tmin > tmax) return false;
    }

    return true;
}

// Closest-approach distance between two moving point-bullets during t∈[0,1].
// Returns true if the squared min distance falls under r2.
static bool bullets_collide(
    float ax0, float ay0, float ax1, float ay1,
    float bx0, float by0, float bx1, float by1,
    float r2
) {
    float p0x = ax0 - bx0, p0y = ay0 - by0;
    float p1x = ax1 - bx1, p1y = ay1 - by1;
    float dx = p1x - p0x, dy = p1y - p0y;
    float aq = dx*dx + dy*dy;
    float bq = p0x*dx + p0y*dy;
    float t = (aq > 1e-8f) ? -bq/aq : 0.0f;
    if (t < 0.0f) t = 0.0f;
    else if (t > 1.0f) t = 1.0f;
    float mx = p0x + t*dx, my = p0y + t*dy;
    return (mx*mx + my*my) < r2;
}

static inline void add_agent_reward(Robocode* env, int agent_idx, float reward) {
    *env->agents[agent_idx].rewards += reward;
    env->logs[agent_idx].episode_return += reward;
}

static inline void record_melee_damage_inflicted(Robocode* env, int agent_idx, float damage) {
    if (damage <= 0.0f || agent_idx < 0 || agent_idx >= env->num_agents) return;
    env->logs[agent_idx].melee_damage_inflicted += damage;
    add_agent_reward(env, agent_idx,
        damage * env->robots[agent_idx].reward_melee_damage_inflicted);
}

static inline void record_damage_taken(Robocode* env, int agent_idx, float damage) {
    if (damage <= 0.0f || agent_idx < 0 || agent_idx >= env->num_agents) return;
    env->logs[agent_idx].damage_taken += damage;
    add_agent_reward(env, agent_idx,
        damage * env->robots[agent_idx].reward_damage_taken);
}

static inline void record_range_damage_inflicted(Robocode* env, int agent_idx, float damage) {
    if (damage <= 0.0f || agent_idx < 0 || agent_idx >= env->num_agents) return;
    env->logs[agent_idx].range_damage_inflicted += damage;
    add_agent_reward(env, agent_idx,
        damage * env->robots[agent_idx].reward_range_damage_inflicted);
}

static inline void record_melee_collision(Robocode* env, int a_idx, int b_idx, float damage) {
    record_melee_damage_inflicted(env, a_idx, damage);
    record_melee_damage_inflicted(env, b_idx, damage);
    record_damage_taken(env, a_idx, damage);
    record_damage_taken(env, b_idx, damage);
}

void move(Robocode* env, Robot* robot, float distance) {
    int robot_idx = (int)(robot - env->robots);
    float dx = cos_deg(robot->heading);
    float dy = sin_deg(robot->heading);
    //float accel = 1.0;//2.0*distance / (robot->v * robot->v);
    float accel = distance;
    float handling = fmaxf(robot->handling_mult, 0.0f);
    float max_speed = 8.0f * fmaxf(robot->speed_mult, 0.0f);

    if (accel > handling) {
        accel = handling;
    } else if (accel < -2.0f * handling) {
        accel = -2.0f * handling;
    }

    robot->v += accel;
    if (robot->v > max_speed) {
        robot->v = max_speed;
    } else if (robot->v < -max_speed) {
        robot->v = -max_speed;
    }

    float new_x = robot->x + dx * robot->v;
    float new_y = robot->y + dy * robot->v;

    // Collision check
    int total_robots = env->num_agents + env->num_bots;
    for (int j = 0; j < total_robots; j++) {
        Robot* target = &env->robots[j];
        if (target == robot) {
            continue;
        }
        if (target->energy < 0) {
            continue;
        }
        float abs_x = fabsf(target->x - new_x);
        float abs_y = fabsf(target->y - new_y);
        if(abs_x > 32.0f || abs_y > 32.0f){
            continue;
        }

        float melee_damage = 0.6f;
        record_melee_collision(env, robot_idx, j, melee_damage);
        target->energy -= melee_damage;
        robot->energy -= melee_damage;
        robot->v = 0;
        target->v = 0;   // both robots stop on ramming collision (classic rule)
        // Mirror bullet kills: +1/-1 terminal reward, perf = bots killed.
        bool s_agent = robot_idx < env->num_agents;
        bool t_agent = j < env->num_agents;
        bool killed = target->energy <= 0.0f;
        if (s_agent && killed) {
            add_agent_reward(env, robot_idx, 1.0f);
            if (!t_agent) env->logs[robot_idx].perf += 1.0f;
        }
        if (t_agent && killed) {
            add_agent_reward(env, j, -1.0f);
        }
        return;
    }
    
    robot->x = new_x;
    robot->y = new_y;

}

float turn(float* heading, float degrees, float d_angle, float turn_offset) {
    if (degrees > d_angle) {
        degrees = d_angle;
    } else if (degrees < -d_angle) {
        degrees = -d_angle;
    }

    *heading += (degrees + turn_offset);
    if (*heading >= 360.0f) {
        *heading -= 360.0f;
    } else if (*heading < 0.0f) {
        *heading += 360.0f;
    }
    return degrees;
}

void fire(Robocode* env, Robot* robot, int robot_idx, float firepower) {
    if (robot->gun_heat > 0) {
        return;
    }
    if (robot->energy < firepower) {
        return;
    }
    robot->energy -= firepower;

    Bullet* bullet = &env->bullets[robot_idx*NUM_BULLETS + robot->bullet_idx];
    robot->bullet_idx = (robot->bullet_idx + 1) % NUM_BULLETS;
    robot->gun_heat += 1.0f + firepower/5.0f;

    bullet->x = robot->x + 16*cos_deg(robot->gun_heading);
    bullet->y = robot->y + 16*sin_deg(robot->gun_heading);
    bullet->heading = robot->gun_heading;
    bullet->firepower = firepower;
    bullet->live = true;
}

static inline float rand_unit(Robocode* env) {
    return (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
}

static inline void sample_dr_triplet(Robocode* env, float* a, float* b, float* c) {
    *a = 1.0f;
    *b = 1.0f;
    *c = 1.0f;

    if (env->dr <= 0.0f) return;
    float upper = 1.0f + env->dr;
    if (upper <= 0.0f) return;
    float lower = 1.0f / upper;
    float width = upper - lower;
    if (width <= 0.0f) return;

    for (int tries = 0; tries < 64; tries++) {
        float first = lower + width * rand_unit(env);
        float second = lower + width * rand_unit(env);
        float third = 3.0f - first - second;
        if (third >= lower && third <= upper) {
            *a = first;
            *b = second;
            *c = third;
            return;
        }
    }
}

static inline void assign_agent_reward_coefficients(Robocode* env, Robot* robot, int agent_idx) {
    if (agent_idx == 0) {
        robot->reward_melee_damage_inflicted = env->reward_melee_damage_inflicted_slot_0;
        robot->reward_damage_taken = env->reward_damage_taken_slot_0;
        robot->reward_range_damage_inflicted = env->reward_range_damage_inflicted_slot_0;
    } else if (agent_idx == 1) {
        robot->reward_melee_damage_inflicted = env->reward_melee_damage_inflicted_slot_1;
        robot->reward_damage_taken = env->reward_damage_taken_slot_1;
        robot->reward_range_damage_inflicted = env->reward_range_damage_inflicted_slot_1;
    } else {
        robot->reward_melee_damage_inflicted = env->reward_melee_damage_inflicted;
        robot->reward_damage_taken = env->reward_damage_taken;
        robot->reward_range_damage_inflicted = env->reward_range_damage_inflicted;
    }
}

int scan_area(Robocode* env, Robot* robot){
    // Sweep is the signed angle traversed from radar_heading_prev to
    // radar_heading, normalized to (-180, 180]. A robot is scanned if its
    // bearing from us lies inside this wedge and within 1200 units.
    float start = robot->radar_heading_prev;
    float sweep = robot->radar_heading - start;
    if (sweep > 180.0f) sweep -= 360.0f;
    if (sweep < -180.0f) sweep += 360.0f;

    int total_robots = env->num_agents + env->num_bots;
    for (int j = 0; j < total_robots; j++) {
        Robot* other = &env->robots[j];
        if (other == robot) continue;
        if (other->energy < 0) continue;

        float dx = other->x - robot->x;
        float dy = other->y - robot->y;
        if (dx*dx + dy*dy > 1200.0f*1200.0f) continue;

        float bearing = atan2f(dy, dx) * 180.0f / 3.14159265358979323846f;
        if (bearing < 0.0f) bearing += 360.0f;

        float diff = bearing - start;
        if (diff > 180.0f) diff -= 360.0f;
        if (diff < -180.0f) diff += 360.0f;

        // diff is inside the wedge when it shares sign with sweep and is shorter.
        if (diff * sweep >= 0.0f && fabsf(diff) <= fabsf(sweep)) return j;
    }
    return -1;
}

void compute_observations(Robocode* env){
    for(int i = 0; i < env->num_agents; i++){
        Robot* robot = &env->robots[i];
        obs_t* obs = env->agents[i].observations;
        obs[0] = robot->x / env->width;
        obs[1] = robot->y / env->height;
        // Absolute headings stored as degrees in [0, 360); convert to radians [0, 2π).
        obs[2] = robot->heading * DEG2RAD;
        obs[3] = robot->gun_heading * DEG2RAD;
        obs[4] = robot->radar_heading * DEG2RAD;
        obs[5] = robot->radar_heading_prev * DEG2RAD;
        obs[6] = robot->v / 8.0f;
        obs[7] = robot->energy / 100.0f;
        obs[8] = robot->speed_mult;
        obs[9] = robot->power_mult;
        obs[10] = robot->handling_mult;
        obs[11] = robot->reward_melee_damage_inflicted;
        obs[12] = robot->reward_damage_taken;
        obs[13] = robot->reward_range_damage_inflicted;

        int scanned = scan_area(env, robot);
        if (scanned < 0) {
            memset(&obs[EGO_FEATURES], 0, OTHER_FEATURES * sizeof(float));
            continue;
        }
        *env->agents[i].rewards += env->reward_spot;
        env->logs[i].episode_return += env->reward_spot;
        // Zero-sum: penalize the scanned agent (being seen = bad).
        // Guarded so bots (j >= num_agents) don't trigger an OOB write.
        if (scanned < env->num_agents) {
            *env->agents[scanned].rewards -= env->reward_spot;
            env->logs[scanned].episode_return -= env->reward_spot;
        }
        Robot* other = &env->robots[scanned];
        // Relative position rotated into ego (body) frame. Engine convention:
        // cos_deg -> x, sin_deg -> y, so forward = (cos h, sin h), right = (sin h, -cos h).
        float dx_w = other->x - robot->x;
        float dy_w = other->y - robot->y;
        float c = cos_deg(robot->heading);
        float s = sin_deg(robot->heading);
        float dx_ego =  c*dx_w + s*dy_w;   // forward
        float dy_ego = -s*dx_w + c*dy_w;   // perpendicular (matches drive's R(-h))
        // Relative headings: wrap raw delta (in (-360, 360)) to (-180, 180]
        // then scale to [-1, 1].
        float dh_body  = other->heading - robot->heading;
        float dh_gun   = other->heading - robot->gun_heading;
        float dh_radar = other->heading - robot->radar_heading;
        if (dh_body  >  180.0f) dh_body  -= 360.0f; else if (dh_body  < -180.0f) dh_body  += 360.0f;
        if (dh_gun   >  180.0f) dh_gun   -= 360.0f; else if (dh_gun   < -180.0f) dh_gun   += 360.0f;
        if (dh_radar >  180.0f) dh_radar -= 360.0f; else if (dh_radar < -180.0f) dh_radar += 360.0f;
        // Aim error: bearing to target (world) minus my gun heading, wrapped to (-180, 180].
        // ~0 means gun is pointed at the target.
        float bearing = atan2f(dy_w, dx_w) * 180.0f / 3.14159265358979323846f;
        float aim_err = bearing - robot->gun_heading;
        if (aim_err >  180.0f) aim_err -= 360.0f;
        else if (aim_err < -180.0f) aim_err += 360.0f;
        int off = EGO_FEATURES;
        obs[off + 0] = dx_ego / 1200.0f;
        obs[off + 1] = dy_ego / 1200.0f;
        obs[off + 2] = dh_body  * DEG2RAD;
        obs[off + 3] = dh_gun   * DEG2RAD;
        obs[off + 4] = dh_radar * DEG2RAD;
        obs[off + 5] = other->energy / 100.0f;
        obs[off + 6] = aim_err * DEG2RAD;
        obs[off + 7] = 1.0f;
    }
}
void puf_reset(Robocode* env) {
    env->tick = 0;
    // boundary_reached is owned by selfplay alignment; do not clear it here.
    int total_robots = env->num_agents + env->num_bots;
    memset(env->bullets, 0, NUM_BULLETS * total_robots * sizeof(Bullet));
    // One DR draw per episode shared by all agents so selfplay slots stay fair
    // (independent per-slot draws can make matches unwinnable).
    float speed_mult = 1.0f, handling_mult = 1.0f, power_mult = 1.0f;
    sample_dr_triplet(env, &speed_mult, &handling_mult, &power_mult);
    int idx = 0;
    float x, y;
    while (idx < total_robots) {
        Robot* robot = &env->robots[idx];
        x = 16 + rand_r(&env->rng) % (env->width-32);
        y = 16 + rand_r(&env->rng) % (env->height-32);
        bool collided = false;
        for (int j = 0; j < idx; j++) {
            Robot* other = &env->robots[j];
            float abs_x = fabsf(x - other->x);
            float abs_y = fabsf(y - other->y);
            if(abs_x <= 32.0f && abs_y <= 32.0f){
                collided = true;
                break;
            }
        }
        if (!collided) {
            robot->x = x;
            robot->y = y;
            robot->v = 0;
            robot->heading = 0;
            robot->gun_heading = 0;
            robot->radar_heading = 0;
            robot->radar_heading_prev = 0;
            robot->energy = 100.0f;
            robot->gun_heat = 3;
            robot->bullet_idx = 0;
            if (idx < env->num_agents) {
                robot->speed_mult = speed_mult;
                robot->handling_mult = handling_mult;
                robot->power_mult = power_mult;
                assign_agent_reward_coefficients(env, robot, idx);
                env->logs[idx] = (Log){0};
            } else {
                robot->speed_mult = 1.0f;
                robot->handling_mult = 1.0f;
                robot->power_mult = 1.0f;
                robot->reward_melee_damage_inflicted = 0.0f;
                robot->reward_damage_taken = 0.0f;
                robot->reward_range_damage_inflicted = 0.0f;
            }
            idx += 1;
        }
    }
    bot_mems_episode_reset(env);
    compute_observations(env);
}

#include "bots.h"

// Agent matches end as soon as a slot reaches zero energy. If all slots are
// disabled in the same tick, score the episode as a draw. Return 2 = no end.
static inline int agent_terminal_outcome(Robocode* env) {
    if (env->num_agents <= 0) return 2;
    bool slot0_dead = env->robots[0].energy <= 0.0f;
    if (env->num_agents == 1) return slot0_dead ? -1 : 2;

    bool any_nonzero_dead = false;
    bool any_nonzero_alive = false;
    for (int a = 1; a < env->num_agents; a++) {
        if (env->robots[a].energy <= 0.0f) any_nonzero_dead = true;
        else any_nonzero_alive = true;
    }

    if (slot0_dead && !any_nonzero_alive) return 0;
    if (slot0_dead) return -1;
    if (any_nonzero_dead) return +1;
    return 2;
}

// Helper for every episode-end path. outcome: +1 slot-0 won, -1 slot-0 lost,
// 0 draw. Historical accounting only applies when env->tag > 0.
static inline void end_episode(Robocode* env, int outcome) {
    float s0_score = (outcome > 0) ? 1.0f : (outcome < 0) ? 0.0f : 0.5f;
    // Noise faced this episode (pre-decay). Bot vs scripted, hist vs frozen, else 0.
    float noise = 0.0f;
    if (env->num_bots > 0 && env->num_agents == 1)
        noise = env->bot_cl_noise;
    else if (env->tag > 0)
        noise = env->hist_cl_noise;
    if (noise < 0.0f) noise = 0.0f;
    if (noise > 1.0f) noise = 1.0f;
    // Scale by num_agents so that (policy_0_score / n) where n increments by
    // num_agents per episode in add_log gives the win rate directly. match()
    // reads this from env/policy_0_score after eval_log divides by n.
    env->log.policy_0_score += s0_score * env->num_agents;
    env->log.policy_1_score += (1.0f - s0_score) * env->num_agents;
    env->log.cl_perf += s0_score * (1.0f - noise) * env->num_agents;
    if (outcome == 0) env->log.draw_rate += env->num_agents;
    if (env->tag > 0) {
        env->boundary_reached = 1;
    }
    // Snapshot pre-decay noise for metrics (what the agent actually faced).
    for (int a = 0; a < env->num_agents; a++) {
        env->logs[a].bot_cl_noise = (env->num_bots > 0 && env->num_agents == 1)
            ? noise : 0.0f;
        env->logs[a].hist_cl_noise = (env->tag > 0) ? noise : 0.0f;
    }
    // Curriculum: primary win → harden bot / frozen opp (lower random rate).
    if (outcome > 0 && env->num_bots > 0 && env->bot_cl_decay > 0.0f
            && env->bot_cl_noise > 0.0f) {
        env->bot_cl_noise -= env->bot_cl_decay;
        if (env->bot_cl_noise < 0.0f) env->bot_cl_noise = 0.0f;
    }
    if (outcome > 0 && env->tag > 0 && env->hist_cl_decay > 0.0f
            && env->hist_cl_noise > 0.0f) {
        env->hist_cl_noise -= env->hist_cl_decay;
        if (env->hist_cl_noise < 0.0f) env->hist_cl_noise = 0.0f;
    }
    for (int a = 0; a < env->num_agents; a++) {
        *env->agents[a].terminals = 1.0f;
    }
    add_log(env);
    puf_reset(env);
}

// Hold Left Shift + WASD/QE/arrows/space.
static void robocode_human_controls(Robocode *env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return;
    }
    float *actions = env->agents[0].actions;
    actions[0] = 2;
    actions[1] = 4;
    actions[2] = 5;
    actions[3] = 5;
    actions[4] = 0;
    if (IsKeyDown(KEY_W)) {
        actions[0] = 3.0f;
    }
    if (IsKeyDown(KEY_S)) {
        actions[0] = 1.0f;
    }
    if (IsKeyDown(KEY_A)) {
        actions[1] = 3.0f;
    }
    if (IsKeyDown(KEY_D)) {
        actions[1] = 5.0f;
    }
    if (IsKeyDown(KEY_Q)) {
        actions[2] = 4.0f;
    }
    if (IsKeyDown(KEY_E)) {
        actions[2] = 6.0f;
    }
    if (IsKeyDown(KEY_LEFT)) {
        actions[3] = 0.0f;
    }
    if (IsKeyDown(KEY_RIGHT)) {
        actions[3] = 8.0f;
    }
    if (IsKeyDown(KEY_SPACE)) {
        actions[4] = 1.0f;
    }
}

static inline void bot_vs_bot_check(Robocode* env) {
    if (env->num_agents > 0 || env->num_bots <= 0) return;
    int alive = 0, winner = -1;
    for (int b = 0; b < env->num_bots; b++) {
        if (env->robots[b].energy > 0.0f) {
            alive++;
            winner = b;
        }
    }
    if (alive == 1) env->bot_match_winner = winner;
    else if (alive == 0) env->bot_match_winner = -1;
}

void puf_step(Robocode* env) {
    // Timeout: all agents step in lockstep, so logs[0].episode_length is shared.
    env->tick += 1;
    if (env->tick > env->max_ticks) {
        if (env->num_agents <= 0) {
            env->bot_match_winner = -1;
            return;
        }
        end_episode(env, 0);  // draw
        return;
    }

    int total_robots = env->num_agents + env->num_bots;
    int total_bullets = total_robots * NUM_BULLETS;

    // Reset per-agent reward/terminal and short-circuit reset on agent death.
    for (int a = 0; a < env->num_agents; a++) {
        *env->agents[a].rewards = 0.0f;
        *env->agents[a].terminals = 0.0f;
    }
    int agent_outcome = agent_terminal_outcome(env);
    if (agent_outcome != 2) {
        end_episode(env, agent_outcome);
        return;
    }
    // move all bullets
    float prev_x[total_bullets], prev_y[total_bullets];
    for (int i = 0; i < total_bullets; i++) {
        Bullet* b = &env->bullets[i];
        if (!b->live) continue;
        prev_x[i] = b->x; prev_y[i] = b->y;
        float v = 20.0f - 3.0f * b->firepower;
        b->x += v * cos_deg(b->heading);
        b->y += v * sin_deg(b->heading);
        if (b->x < 0 || b->x > env->width || b->y < 0 || b->y > env->height) {
            b->live = false;
        }
    }

    // bullet-bullet collisions. 
    for (int i = 0; i < total_bullets; i++) {
        if (!env->bullets[i].live) continue;
        for (int j = i + 1; j < total_bullets; j++) {
            if (!env->bullets[j].live) continue;
            if (bullets_collide(prev_x[i], prev_y[i], env->bullets[i].x, env->bullets[i].y,
                                prev_x[j], prev_y[j], env->bullets[j].x, env->bullets[j].y,
                                64.0f)) {
                env->bullets[i].live = false;
                env->bullets[j].live = false;
                break;
            }
        }
    }

    // bullet-robot collisions + rewards.
    bool any_bot_alive = (env->num_bots == 0);
    for (int shooter = 0; shooter < total_robots; shooter++) {
        Robot* robot = &env->robots[shooter];
        if (shooter >= env->num_agents && robot->energy > 0.0f) any_bot_alive = true;
        for (int blt = 0; blt < NUM_BULLETS; blt++) {
            int bi = shooter * NUM_BULLETS + blt;
            Bullet* bullet = &env->bullets[bi];
            if (!bullet->live) continue;

            for (int j = 0; j < total_robots; j++) {
                if (!bullet->live) break;  
                if (j == shooter) continue;
                Robot* target = &env->robots[j];
                if (target->energy < 0) continue;
                // Broad-phase: keep if EITHER endpoint of the swept segment is
                // within 32 units of target center. Using only the end position
                // would miss tunneling at firepower 0.1 (speed ~19.7/tick).
                float dx0 = target->x - prev_x[bi],   dy0 = target->y - prev_y[bi];
                float dx1 = target->x - bullet->x,   dy1 = target->y - bullet->y;
                float d2  = fminf(dx0*dx0 + dy0*dy0, dx1*dx1 + dy1*dy1);
                if (d2 > 1024.0f) continue;
                bool hit = segment_intersects_aabb(
                        prev_x[bi], prev_y[bi],
                        bullet->x, bullet->y,
                        target->x - 16, target->x + 16,
                        target->y - 16, target->y + 16
                );
                if (!hit) continue;

                float damage = 4 * bullet->firepower;
                if (bullet->firepower > 1.0f) damage += 2*(bullet->firepower - 1.0f);

                target->energy -= damage;
                robot->energy += 3 * bullet->firepower;
                bullet->live = false;

                bool s_agent = shooter < env->num_agents;
                bool t_agent = j < env->num_agents;
                if (!t_agent && s_agent) {
                    bot_on_hit_by_bullet(env, j, bullet->heading, bullet->firepower);
                }
                // DrussGT gun learning when a bot's bullet hits anyone.
                if (!s_agent && env->bot_mems != NULL
                        && bot_policy_for(env, shooter) == BOT_DRUSSGT) {
                    int bmem = shooter - env->num_agents;
                    if (bmem >= 0 && bmem < env->num_bots) {
                        dgt_on_bullet_hit(&env->bot_mems[bmem],
                                          target->x, target->y);
                    }
                }
                bool killed = target->energy <= 0.0f;
                if (s_agent) {
                    record_range_damage_inflicted(env, shooter, damage);
                    env->logs[shooter].score += damage;
                    if (killed) add_agent_reward(env, shooter, 1.0f);
                    if (killed && !t_agent) env->logs[shooter].perf += 1.0f;
                }
                if (t_agent) {
                    record_damage_taken(env, j, damage);
                    if (killed) add_agent_reward(env, j, -1.0f);
                }
            }
        }
    }
    if (env->num_bots > 0 && !any_bot_alive) {
        if (env->num_agents <= 0) {
            env->bot_match_winner = -1;
            return;
        }
        end_episode(env, +1);  // primary wiped all bots
        return;
    }
    bot_vs_bot_check(env);
    if (env->num_agents <= 0 && env->bot_match_winner != -2) return;
    agent_outcome = agent_terminal_outcome(env);
    if (agent_outcome != 2) {
        end_episode(env, agent_outcome);
        return;
    }
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        float* atn = env->agents[i].actions;
        env->logs[i].episode_length += 1.0f;

        // Defensive guard; agent_terminal_outcome should have already ended
        // episodes for disabled agents.
        if (robot->energy <= 0.0f) {
            robot->v = 0;
            continue;
        }

        // Hist curriculum: with hist_cl_noise, replace frozen slot actions.
        if (i > 0 && env->tag > 0 && env->hist_cl_noise > 0.0f
                && rand_unit(env) < env->hist_cl_noise) {
            atn[0] = (float)(rand_r(&env->rng) % 4);
            atn[1] = (float)(rand_r(&env->rng) % 9);
            atn[2] = (float)(rand_r(&env->rng) % 11);
            atn[3] = (float)(rand_r(&env->rng) % 11);
            atn[4] = (float)(rand_r(&env->rng) % 6);
        }

        // Cool down gun
        if (robot->gun_heat > 0) {
            robot->gun_heat -= 0.1f;
        }

        // Move
        float handling = fmaxf(robot->handling_mult, 0.0f);
        float move_atn = ACCEL_VALUES[(int)atn[0]] * handling;
        move(env, robot, move_atn);

        // Turn
        float turn_atn = TURN_VALUES[(int)atn[1]] * handling;

        float abs_v = fabs(robot->v);
        float max_turn = (10 - 0.75*abs_v) * handling;
        if (max_turn < 0.0f) max_turn = 0.0f;
        float body_turn_degrees = turn(&robot->heading, turn_atn, max_turn, 0);

        // Gun
        float gun_atn = GUN_TURN_VALUES[(int)atn[2]] * handling;
        float gun_degrees = turn(&robot->gun_heading, gun_atn, 20.0f * handling, body_turn_degrees);

        // Radar
        float radar_atn = RADAR_TURN_VALUES[(int)atn[3]] * handling;
        robot->radar_heading_prev = robot->radar_heading;
        turn(&robot->radar_heading, radar_atn, 45.0f * handling, body_turn_degrees+gun_degrees);

        // Fire
        float firepower = FIREPOWER_VALUES[(int)atn[4]] * fmaxf(robot->power_mult, 0.0f);
        if (firepower > 0) {
            fire(env, robot,i, firepower);
        }

        // Clip position
        float px = robot->x, py = robot->y;
        robot->x = fmaxf(16.0f, fminf(robot->x, env->width  - 16.0f));
        robot->y = fmaxf(16.0f, fminf(robot->y, env->height - 16.0f));
        int hit_wall = (robot->x != px) || (robot->y != py);
        
        // Damage from wall collisions & stop bot
        if(!hit_wall) continue;
        float wall_dmg = fabsf(robot->v)*0.5 - 1;
        if (wall_dmg < 0.0f){
            wall_dmg = 0.0f;
        }
        robot->energy -= wall_dmg;
        record_damage_taken(env, i, wall_dmg);
        robot->v = 0;
    }
    agent_outcome = agent_terminal_outcome(env);
    if (agent_outcome != 2) {
        end_episode(env, agent_outcome);
        return;
    }

    // bot step
    for (int b = env->num_agents; b < total_robots; b++) bot_step(env, b);
    if (env->num_bots > 0) {
        any_bot_alive = false;
        for (int b = env->num_agents; b < total_robots; b++) {
            if (env->robots[b].energy > 0.0f) {
                any_bot_alive = true;
                break;
            }
        }
        if (!any_bot_alive) {
            if (env->num_agents <= 0) {
                env->bot_match_winner = -1;
                return;
            }
            end_episode(env, +1);
            return;
        }
        bot_vs_bot_check(env);
        if (env->num_agents <= 0 && env->bot_match_winner != -2) return;
    }
    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    Texture2D atlas;
};

Client* make_client(Robocode* env) {
    InitWindow(env->width, env->height, "PufferLib Ray Robocode");
    SetTargetFPS(60);
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->atlas = LoadTexture("resources/robocode/robocode.png");
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->atlas);
    CloseWindow();
}

void puf_render(Robocode* env) {
    if(env->client == NULL){
        env->client = make_client(env);
    }
    robocode_human_controls(env);
    Client* client = env->client;
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int x = 0; x < env->width; x+=64) {
        for (int y = 0; y < env->height; y+=64) {
            int src_x = 64 * ((x*33409 + y*30971) % 5);
            int w = (x + 64 > env->width)  ? env->width  - x : 64;
            int h = (y + 64 > env->height) ? env->height - y : 64;
            Rectangle src_rect = (Rectangle){src_x, 0, w, h};
            Rectangle dst_rect = (Rectangle){x, y, w, h};
            DrawTexturePro(client->atlas, src_rect, dst_rect, (Vector2){0, 0}, 0, WHITE);
        }
    }

    int total_robots = env->num_agents + env->num_bots;
    for (int i = 0; i < total_robots; i++) {
        Robot robot = env->robots[i];
        if (robot.energy < 0) continue;
        bool is_agent = i < env->num_agents;
        Vector2 robot_pos = (Vector2){robot.x, robot.y};

        // Radar wedge: agents = green, bots = orange. Use the actual radar
        // sweep sign so vertex winding stays consistent (no backface culling).
        float sweep = robot.radar_heading - robot.radar_heading_prev;
        if (sweep >  180.0f) sweep -= 360.0f;
        else if (sweep < -180.0f) sweep += 360.0f;
        float a_left  = (sweep >= 0) ? robot.radar_heading      : robot.radar_heading_prev;
        float a_right = (sweep >= 0) ? robot.radar_heading_prev : robot.radar_heading;
        Vector2 p_left  = (Vector2){robot.x + 1200*cos_deg(a_left),  robot.y + 1200*sin_deg(a_left)};
        Vector2 p_right = (Vector2){robot.x + 1200*cos_deg(a_right), robot.y + 1200*sin_deg(a_right)};
        Color wedge_color = is_agent ? (Color){0, 255, 0, 128} : (Color){255, 140, 0, 128};
        DrawTriangle(robot_pos, p_left, p_right, wedge_color);

        int src_y = is_agent ? 64 : 128;  // blue row for agents, red row for bots
        Rectangle body_rect  = (Rectangle){0,   src_y, 64, 64};
        Rectangle radar_rect = (Rectangle){64,  src_y, 64, 64};
        Rectangle gun_rect   = (Rectangle){128, src_y, 64, 64};
        Rectangle dest_rect  = (Rectangle){robot.x, robot.y, 64, 64};
        Vector2 origin = (Vector2){32, 32};
        DrawTexturePro(client->atlas, body_rect,  dest_rect, origin, robot.heading+90,       WHITE);
        DrawTexturePro(client->atlas, radar_rect, dest_rect, origin, robot.radar_heading+90, WHITE);
        DrawTexturePro(client->atlas, gun_rect,   dest_rect, origin, robot.gun_heading+90,   WHITE);

        DrawText(TextFormat("%.1f", robot.energy), robot.x-16, robot.y-48, 12, WHITE);
    }

    for (int i = 0; i < (env->num_agents + env->num_bots)*NUM_BULLETS; i++) {
        Bullet bullet = env->bullets[i];
        if (!bullet.live) {
            continue;
        }
        Vector2 bullet_pos = (Vector2){bullet.x, bullet.y};
        DrawCircleV(bullet_pos, 4, WHITE);
    }

    const char* tick_text = TextFormat("%i", env->tick);
    DrawText(tick_text, 10, 10, 10, WHITE);

    EndDrawing();
    puf_web_vsync();
}
