#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"

#define NUM_ACTIONS 5
#define NUM_BULLETS 16
#define EGO_FEATURES 8
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
    return cos(deg * 3.14159265358979323846 / 180.0);
}

float sin_deg(float deg) {
    return sin(deg * 3.14159265358979323846 / 180.0);
}

typedef struct BotMem BotMem;  // defined in bots.h

#define ROBOCODE_MAX_BANKS 2

typedef struct Log Log;
struct Log {
    float perf;             // bots killed this episode
    float episode_return;
    float episode_length;
    float score;            // damage dealt this episode
    float damage_received;  // starting energy - current energy at episode end
    // Historical pool tracking (selfplay-pool mode). Per-bank score/games for
    // matches against frozen historical opponents. hist_score / hist_n are
    // legacy aggregates summed across all banks.
    float hist_score;
    float hist_n;
    float hist_score_bank[ROBOCODE_MAX_BANKS];
    float hist_n_bank[ROBOCODE_MAX_BANKS];
    // Per-slot scores for match() scoring + selfplay sanity-check. In selfplay
    // both should average to ~0.5; in match A=slot 0, B=slot 1, slot_0_score is
    // the win rate of policy A. Each game contributes 1.0 worth of credit total
    // (win=1.0 to winner, draw=0.5 each). We scale by num_agents on accumulation
    // so the eval_log mean (sum / n where n increments by num_agents per ep)
    // equals win_rate directly.
    float slot_0_score;
    float slot_1_score;
    float draw_rate;
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
    int bullet_idx;
    float gun_heat;
    int energy;
};

typedef struct Client Client;
typedef struct Robocode Robocode;
struct Robocode {
    Client* client;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    // Per-slot pointers populated by my_setup_perm (MY_USES_PERM). Required for
    // selfplay-pool mode where agent_perm reroutes logical slots into specific
    // physical rows (primary vs frozen-bank). For non-selfplay (num_agents=1)
    // these still point at the env's single slot.
    float* obs_ptr[2];
    float* action_ptr[2];
    float* reward_ptr[2];
    float* terminal_ptr[2];

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
    float reward_damage;
    float reward_spot;
    int bot_policy;
    BotMem* bot_mems;        // per-bot scratch (allocated by bots.h)

    // Selfplay-pool tagging. tag = 0 means pure selfplay (both slots = primary
    // policy). tag = 1..ROBOCODE_MAX_BANKS means historical: slot 0 = primary,
    // slot 1 = frozen historical opponent from bank (tag - 1). boundary_reached
    // is set on game-end so Python can detect when historical envs have all
    // completed at least one game since the last swap arm.
    int tag;
    int boundary_reached;

    unsigned int rng;
};

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

void allocate_env(Robocode* env) {
    int obs_size = EGO_FEATURES + OTHER_FEATURES;
    init(env);
    env->observations =(float*)calloc(obs_size*env->num_agents, sizeof(float));
    env->actions = (float*)calloc(NUM_ACTIONS*env->num_agents, sizeof(float));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->terminals = (float*)calloc(env->num_agents, sizeof(float));
    // Standalone (non-vecenv) path: wire per-slot pointers to adjacent rows of
    // the env-owned buffers. vecenv path overrides these via my_setup_perm.
    for (int s = 0; s < env->num_agents; s++) {
        env->obs_ptr[s]      = env->observations + s * obs_size;
        env->action_ptr[s]   = env->actions + s * NUM_ACTIONS;
        env->reward_ptr[s]   = env->rewards + s;
        env->terminal_ptr[s] = env->terminals + s;
    }
}


void c_close(Robocode* env) {
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
        env->log.score           += env->logs[i].score;
        env->log.damage_received += env->logs[i].damage_received;
        env->log.n               += 1.0f;
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

void move(Robocode* env, Robot* robot, float distance) {
    float dx = cos_deg(robot->heading);
    float dy = sin_deg(robot->heading);
    //float accel = 1.0;//2.0*distance / (robot->v * robot->v);
    float accel = distance;

    if (accel > 1.0) {
        accel = 1.0;
    } else if (accel < -2.0) {
        accel = -2.0;
    }

    robot->v += accel;
    if (robot->v > 8.0) {
        robot->v = 8.0;
    } else if (robot->v < -8.0) {
        robot->v = -8.0;
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
        if (target->energy < 0) {     // dead = phase-through; disabled = still solid
            continue;
        }
        float abs_x = fabsf(target->x - new_x);
        float abs_y = fabsf(target->y - new_y);
        if(abs_x > 32.0f || abs_y > 32.0f){
            continue;
        }

        target->energy -= 0.6;
        robot->energy -= 0.6;
        robot->v = 0;
        target->v = 0;   // both robots stop on ramming collision (classic rule)
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
        if (other->energy < 0) continue;   // disabled (energy=0) is still scannable

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
        float* obs = env->obs_ptr[i];
        obs[0] = robot->x / env->width;
        obs[1] = robot->y / env->height;
        // Absolute headings stored as degrees in [0, 360); convert to radians [0, 2π).
        obs[2] = robot->heading * DEG2RAD;
        obs[3] = robot->gun_heading * DEG2RAD;
        obs[4] = robot->radar_heading * DEG2RAD;
        obs[5] = robot->radar_heading_prev * DEG2RAD;
        obs[6] = robot->v / 8.0f;
        obs[7] = robot->energy / 100.0f;

        int scanned = scan_area(env, robot);
        if (scanned < 0) {
            memset(&obs[EGO_FEATURES], 0, OTHER_FEATURES * sizeof(float));
            continue;
        }
        *env->reward_ptr[i] += env->reward_spot;
        env->logs[i].episode_return += env->reward_spot;
        // Zero-sum: penalize the scanned agent (being seen = bad).
        // Guarded so bots (j >= num_agents) don't trigger an OOB write.
        if (scanned < env->num_agents) {
            *env->reward_ptr[scanned] -= env->reward_spot;
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
        obs[8]  = dx_ego / 1200.0f;
        obs[9]  = dy_ego / 1200.0f;
        obs[10] = dh_body  * DEG2RAD;
        obs[11] = dh_gun   * DEG2RAD;
        obs[12] = dh_radar * DEG2RAD;
        obs[13] = other->energy / 100.0f;
        obs[14] = aim_err * DEG2RAD;
        obs[15] = 1.0f;
    }
}
void c_reset(Robocode* env) {
    env->tick = 0;
    env->boundary_reached = 0;   // cleared so next episode-end can flip it back
    int total_robots = env->num_agents + env->num_bots;
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
            robot->energy = 100;
            robot->gun_heat = 3;
            robot->bullet_idx = 0;
            if (idx < env->num_agents) {
                env->logs[idx] = (Log){0};
            }
            idx += 1;
        }
    }
    bot_mems_episode_reset(env);
    compute_observations(env);
}

#include "bots.h"

// Helper for every episode-end path. outcome: +1 slot-0 won, -1 slot-0 lost,
// 0 draw (timeout). Historical accounting only applies when env->tag > 0.
static inline void end_episode(Robocode* env, int outcome) {
    float s0_score = (outcome > 0) ? 1.0f : (outcome < 0) ? 0.0f : 0.5f;
    // Scale by num_agents so that (slot_0_score / n) where n increments by
    // num_agents per episode in add_log gives the win rate directly. match()
    // reads this from env/slot_0_score after eval_log divides by n.
    env->log.slot_0_score += s0_score * env->num_agents;
    env->log.slot_1_score += (1.0f - s0_score) * env->num_agents;
    if (outcome == 0) env->log.draw_rate += env->num_agents;
    if (env->tag > 0 && env->tag <= ROBOCODE_MAX_BANKS) {
        int bank_idx = env->tag - 1;
        env->log.hist_score_bank[bank_idx] += s0_score;
        env->log.hist_n_bank[bank_idx]     += 1.0f;
        env->log.hist_score                += s0_score;
        env->log.hist_n                    += 1.0f;
        env->boundary_reached = 1;
    }
    for (int a = 0; a < env->num_agents; a++) *env->terminal_ptr[a] = 1.0f;
    add_log(env);
    c_reset(env);
}

void c_step(Robocode* env) {
    // Timeout: all agents step in lockstep, so logs[0].episode_length is shared.
    env->tick += 1;
    if (env->tick > env->max_ticks) {
        end_episode(env, 0);  // draw
        return;
    }

    int total_robots = env->num_agents + env->num_bots;
    int total_bullets = total_robots * NUM_BULLETS;

    // Reset per-agent reward/terminal and short-circuit reset on agent death.
    for (int a = 0; a < env->num_agents; a++) {
        *env->reward_ptr[a]   = 0.0f;
        *env->terminal_ptr[a] = 0.0f;
    }
    for (int a = 0; a < env->num_agents; a++) {
        if (env->robots[a].energy < 0) {  // strict: death requires energy past 0
            // Primary (slot 0) wins iff a non-zero slot died.
            end_episode(env, (a == 0) ? -1 : +1);
            return;
        }
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
        if (shooter >= env->num_agents && robot->energy >= 0) any_bot_alive = true;  // disabled counts as alive
        for (int blt = 0; blt < NUM_BULLETS; blt++) {
            int bi = shooter * NUM_BULLETS + blt;
            Bullet* bullet = &env->bullets[bi];
            if (!bullet->live) continue;

            for (int j = 0; j < total_robots; j++) {
                if (!bullet->live) break;  
                if (j == shooter) continue;
                Robot* target = &env->robots[j];
                if (target->energy < 0) continue;   // disabled (=0) is still hittable
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
                bool killed = target->energy < 0;   // strict: brought past 0
                float r = killed ? 1.0f : damage * env->reward_damage;
                if (s_agent) {
                    *env->reward_ptr[shooter] += r;
                    env->logs[shooter].score += damage;
                    env->logs[shooter].episode_return += r;
                    if (killed && !t_agent) env->logs[shooter].perf += 1.0f;
                }
                if (t_agent) {
                    *env->reward_ptr[j] -= r;
                    env->logs[j].episode_return -= r;
                }
            }
        }
    }
    if (env->num_bots > 0 && !any_bot_alive) {
        end_episode(env, +1);  // primary wiped all bots
        return;
    }
    for (int i = 0; i < env->num_agents; i++) {
        Robot* robot = &env->robots[i];
        float* atn = env->action_ptr[i];
        env->logs[i].episode_length += 1.0f;

        // Disabled (energy <= 0): no actions, velocity frozen to 0.
        // Classic Robocode rule — alive but inert, can still be hit and killed.
        if (robot->energy <= 0) {
            robot->v = 0;
            continue;
        }

        // Cool down gun
        if (robot->gun_heat > 0) {
            robot->gun_heat -= 0.1f;
        }

        // Move
        float move_atn = ACCEL_VALUES[(int)atn[0]];
        move(env, robot, move_atn);

        // Turn
        float turn_atn = TURN_VALUES[(int)atn[1]];

        float abs_v = fabs(robot->v);
        float max_turn = 10 - 0.75*abs_v;
        float body_turn_degrees = turn(&robot->heading, turn_atn, max_turn, 0);

        // Gun
        float gun_atn = GUN_TURN_VALUES[(int)atn[2]];
        float gun_degrees = turn(&robot->gun_heading, gun_atn, 20.0, body_turn_degrees);

        // Radar
        float radar_atn = RADAR_TURN_VALUES[(int)atn[3]];
        robot->radar_heading_prev = robot->radar_heading;
        turn(&robot->radar_heading, radar_atn, 45.0, body_turn_degrees+gun_degrees);

        // Fire
        float firepower = FIREPOWER_VALUES[(int)atn[4]];
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
        robot->v = 0;
    }
    // bot step
    for (int b = env->num_agents; b < total_robots; b++) bot_step(env, b);
    compute_observations(env);
}

typedef struct Client Client;
struct Client {
    Texture2D atlas;
};

Client* make_client(Robocode* env) {
    InitWindow(env->width+100, env->height+100, "PufferLib Ray Robocode");
    SetTargetFPS(60);
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->atlas = LoadTexture("resources/robocode/robocode.png");
    return client;
}

void close_client(Client* client) {
    UnloadTexture(client->atlas);
    CloseWindow();
}

void c_render(Robocode* env) {
    if(env->client == NULL){
        env->client = make_client(env);
    }
    Client* client = env->client;
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    // Center the world inside the padded window.
    Camera2D camera = {0};
    camera.offset = (Vector2){(GetScreenWidth() - env->width)/2.0f,
                              (GetScreenHeight() - env->height)/2.0f};
    camera.zoom = 1.0f;
    BeginMode2D(camera);

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
        if (robot.energy < 0) continue;     // still draw disabled (energy=0) bots
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

        DrawText(TextFormat("%i", robot.energy), robot.x-16, robot.y-48, 12, WHITE);
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

    EndMode2D();
    EndDrawing();
}
