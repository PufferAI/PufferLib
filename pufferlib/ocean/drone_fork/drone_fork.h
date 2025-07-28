// Standalone C demo for DroneSwarm environment
// Compile using: ./scripts/build_ocean.sh drone [local|fast]
// Run with: ./drone


// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/stmio/drone

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"
#include "dronelib.h"

#define TASK_IDLE 0
#define TASK_HOVER 1
#define TASK_ORBIT 2
#define TASK_FOLLOW 3
#define TASK_CUBE 4
#define TASK_CONGO 5
#define TASK_FLAG 6
#define TASK_RACE 7
#define TASK_N 8
#define MAX_ROCKETS 100
#define ROCKET_DMG 100
#define ROCKET_VEL 15.0f

#define HIT_REWARD 5.f
#define HIT_PUNISH 7.f
#define DEATH_PUNISH 10.f


char* TASK_NAMES[TASK_N] = {
    "Idle", "Hover", "Orbit", "Follow",
    "Cube", "Congo", "FLAG", "Race"
};

#define R (Color){255, 0, 0, 255}
#define W (Color){255, 255, 255, 255}
#define B (Color){0, 0, 255, 255}
Color FLAG_COLORS[64] = {
    B, B, B, B, R, R, R, R,
    B, B, B, B, W, W, W, W,
    B, B, B, B, R, R, R, R,
    B, B, B, B, W, W, W, W,
    R, R, R, R, R, R, R, R,
    W, W, W, W, W, W, W, W,
    R, R, R, R, R, R, R, R,
    W, W, W, W, W, W, W, W
};
#undef R
#undef W
#undef B

typedef struct Client Client;
struct Client {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;

    // Trailing path buffer (for rendering only)
    Trail* trails;
};


typedef struct {
    Vec3 pos;
    Vec3 vel;
    Drone* parent;
} Rocket;

int uid_tracker = 0;

typedef struct {
    float *observations;
    float *actions;
    float *rewards;
    unsigned char *terminals;

    Log log;
    int tick;
    int report_interval;

    int task;
    int num_agents;
    Drone* agents;

    Rocket rockets[MAX_ROCKETS];
    int rocket_count;

    int max_rings;
    Ring* ring_buffer;

    Client *client;
} DroneSwarm;
static bool player_active = false;
static int player_idx = -1;
static float player_yaw = 0.0f, player_pitch = 0.0f;
static void player_character(DroneSwarm* env, int idx) {
    Drone *player = &env->agents[idx];
    float *atn = &env->actions[7 * idx];
    const float base = 0.5f;
    float a0 = base, a1 = base, a2 = base, a3 = base;
    const float tilt = 0.2f;
    if (IsKeyDown(KEY_W)) { a0 -= tilt; a1 += tilt; }
    if (IsKeyDown(KEY_S)) { a0 += tilt; a1 -= tilt; }
    if (IsKeyDown(KEY_D)) { a2 += tilt; a3 -= tilt; }
    if (IsKeyDown(KEY_A)) { a2 -= tilt; a3 += tilt; }
    a0 = clampf(a0, 0.0f, 1.0f);
    a1 = clampf(a1, 0.0f, 1.0f);
    a2 = clampf(a2, 0.0f, 1.0f);
    a3 = clampf(a3, 0.0f, 1.0f);
    atn[0] = a0; atn[1] = a1; atn[2] = a2; atn[3] = a3;
    Vector2 md = GetMouseDelta();
    const float sens = 0.003f;
    player_yaw += md.x * sens;
    player_pitch -= md.y * sens;
    if (player_pitch > M_PI/2 - 0.1f) player_pitch = M_PI/2 - 0.1f;
    if (player_pitch < -M_PI/2 + 0.1f) player_pitch = -M_PI/2 + 0.1f;
    Vec3 fwd = {cosf(player_pitch)*cosf(player_yaw), cosf(player_pitch)*sinf(player_yaw), sinf(player_pitch)};
    atn[4] = IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? 1.0f : 0.0f;
    atn[5] = player_yaw / M_PI;
    atn[6] = player_pitch / (M_PI/2);
}

void init(DroneSwarm *env) {
    env->agents = calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = calloc(env->max_rings, sizeof(Ring));
    env->log = (Log){0};
    env->tick = 0;
}

void add_log(DroneSwarm *env, int idx, bool oob) {
    Drone *agent = &env->agents[idx];
    env->log.score += agent->score;
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collision_rate += agent->collisions / (float)agent->episode_length;
    env->log.perf += agent->score / (float)agent->episode_length;
    if (oob) {
        env->log.oob += 1.0f;
    }
    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
}

Drone* nearest_drone(DroneSwarm* env, Drone *agent) {
    float min_dist = 999999.0f;
    Drone *nearest = NULL;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *other = &env->agents[i];
        if (other == agent) {
            continue;
        }
        float dx = agent->pos.x - other->pos.x;
        float dy = agent->pos.y - other->pos.y;
        float dz = agent->pos.z - other->pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = other;
        }
    }
    //if (nearest == NULL) {
      //  int x = 0;
    //}
    return nearest;
}

void compute_observations(DroneSwarm *env) {
    int idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        Quat q_inv = quat_inverse(agent->quat);
        Vec3 linear_vel_body = quat_rotate(q_inv, agent->vel);
        Vec3 drone_up_world = quat_rotate(agent->quat, (Vec3){0.0f, 0.0f, 1.0f});

        // TODO: Need abs observations now right?
        env->observations[idx++] = linear_vel_body.x / agent->max_vel;
        env->observations[idx++] = linear_vel_body.y / agent->max_vel;
        env->observations[idx++] = linear_vel_body.z / agent->max_vel;

        env->observations[idx++] = agent->omega.x / agent->max_omega;
        env->observations[idx++] = agent->omega.y / agent->max_omega;
        env->observations[idx++] = agent->omega.z / agent->max_omega;

        env->observations[idx++] = drone_up_world.x;
        env->observations[idx++] = drone_up_world.y;
        env->observations[idx++] = drone_up_world.z;

        env->observations[idx++] = agent->quat.w;
        env->observations[idx++] = agent->quat.x;
        env->observations[idx++] = agent->quat.y;
        env->observations[idx++] = agent->quat.z;

        env->observations[idx++] = agent->rpms[0] / agent->max_rpm;
        env->observations[idx++] = agent->rpms[1] / agent->max_rpm;
        env->observations[idx++] = agent->rpms[2] / agent->max_rpm;
        env->observations[idx++] = agent->rpms[3] / agent->max_rpm;

        env->observations[idx++] = agent->pos.x / GRID_X;
        env->observations[idx++] = agent->pos.y / GRID_Y;
        env->observations[idx++] = agent->pos.z / GRID_Z;

        env->observations[idx++] = agent->spawn_pos.x / GRID_X;
        env->observations[idx++] = agent->spawn_pos.y / GRID_Y;
        env->observations[idx++] = agent->spawn_pos.z / GRID_Z;

        float dx = agent->target_pos.x - agent->pos.x;
        float dy = agent->target_pos.y - agent->pos.y;
        float dz = agent->target_pos.z - agent->pos.z;
        env->observations[idx++] = clampf(dx, -1.0f, 1.0f);
        env->observations[idx++] = clampf(dy, -1.0f, 1.0f);
        env->observations[idx++] = clampf(dz, -1.0f, 1.0f);
        env->observations[idx++] = dx / GRID_X;
        env->observations[idx++] = dy / GRID_Y;
        env->observations[idx++] = dz / GRID_Z;

        env->observations[idx++] = agent->last_collision_reward;
        env->observations[idx++] = agent->last_target_reward;
        env->observations[idx++] = agent->last_abs_reward;

        // Multiagent obs
        Drone* nearest = nearest_drone(env, agent);
        if (env->num_agents > 1) {
            env->observations[idx++] = clampf(nearest->pos.x - agent->pos.x, -1.0f, 1.0f);
            env->observations[idx++] = clampf(nearest->pos.y - agent->pos.y, -1.0f, 1.0f);
            env->observations[idx++] = clampf(nearest->pos.z - agent->pos.z, -1.0f, 1.0f);
        } else {
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
        }

        // Rocket obs (replace legacy ring features)
        float cooldown_norm = (float)agent->rocket_cooldown / (float)ROCKET_COOLDOWN;
        float rockets_norm = (float)env->rocket_count / (float)MAX_ROCKETS;
        Vec3 dir_body = {0.0f, 0.0f, 0.0f};
        float dist_norm = 0.0f;
        float nearest_dist = MAX_DIST;
        for (int r = 0; r < env->rocket_count; r++) {
            Rocket *rocket = &env->rockets[r];
            if (rocket->parent == agent) continue;
            float dx = rocket->pos.x - agent->pos.x;
            float dy = rocket->pos.y - agent->pos.y;
            float dz = rocket->pos.z - agent->pos.z;
            float d = sqrtf(dx*dx + dy*dy + dz*dz);
            if (d < nearest_dist) {
                nearest_dist = d;
                Vec3 dir_world = {dx, dy, dz};
                if (d > 0) {
                    float inv_d = 1.0f / d;
                    dir_world.x *= inv_d;
                    dir_world.y *= inv_d;
                    dir_world.z *= inv_d;
                }
                dir_body = quat_rotate(q_inv, dir_world);
            }
        }
        if (nearest_dist < MAX_DIST) dist_norm = nearest_dist / MAX_DIST;
        env->observations[idx++] = cooldown_norm;
        env->observations[idx++] = rockets_norm;
        env->observations[idx++] = dir_body.x;
        env->observations[idx++] = dir_body.y;
        env->observations[idx++] = dir_body.z;
        env->observations[idx++] = dist_norm;
    }
    int total = env->num_agents * 41;
    for (int j = 0; j < total; j++) {
        if (!isfinite(env->observations[j])) env->observations[j] = 0.0f;
    }
}

void move_target(DroneSwarm* env, Drone *agent) {
    agent->target_pos.x += agent->target_vel.x;
    agent->target_pos.y += agent->target_vel.y;
    agent->target_pos.z += agent->target_vel.z;
    if (agent->target_pos.x < -GRID_X || agent->target_pos.x > GRID_X) {
        agent->target_vel.x = -agent->target_vel.x;
    }
    if (agent->target_pos.y < -GRID_Y || agent->target_pos.y > GRID_Y) {
        agent->target_vel.y = -agent->target_vel.y;
    }
    if (agent->target_pos.z < -GRID_Z || agent->target_pos.z > GRID_Z) {
        agent->target_vel.z = -agent->target_vel.z;
    }
}

void set_target_idle(DroneSwarm* env, int idx) {
    Drone *agent = &env->agents[idx];
    agent->target_pos = (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), rndf(-MARGIN_Z, MARGIN_Z)};
    agent->target_vel = (Vec3){rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET)};
}

void set_target_hover(DroneSwarm* env, int idx) {
    Drone *agent = &env->agents[idx];
    agent->target_pos = agent->pos;
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_orbit(DroneSwarm* env, int idx) {
    // Fibbonacci sphere algorithm
    float R = 8.0f;
    float phi = PI * (sqrt(5.0f) - 1.0f);
    float y = 1.0f - 2*((float)idx / (float)env->num_agents);
    float radius = sqrtf(1.0f - y*y);

    float theta = phi * idx;

    float x = cos(theta) * radius;
    float z = sin(theta) * radius;

    Drone *agent = &env->agents[idx];
    agent->target_pos = (Vec3){R*x, R*z, R*y}; // convert to z up 
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_follow(DroneSwarm* env, int idx) {
    Drone* agent = &env->agents[idx];
    if (idx == 0) {
        set_target_idle(env, idx);
    } else {
        agent->target_pos = env->agents[0].target_pos;
        agent->target_vel = env->agents[0].target_vel;
    }
}

void set_target_cube(DroneSwarm* env, int idx) {
    Drone* agent = &env->agents[idx];
    float z = idx / 16;
    idx = idx % 16;
    float x = (float)(idx % 4);
    float y = (float)(idx / 4);
    agent->target_pos = (Vec3){4*x - 6, 4*y - 6, 4*z - 6};
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_congo(DroneSwarm* env, int idx) {
    if (idx == 0) {
        set_target_idle(env, idx);
        return;
    }
    Drone* follow = &env->agents[idx - 1];
    Drone* lead = &env->agents[idx];
    lead->target_pos = follow->target_pos;
    lead->target_vel = follow->target_vel;

    // TODO: Slow hack
    for (int i = 0; i < 40; i++) {
        move_target(env, lead);
    }
}

void set_target_flag(DroneSwarm* env, int idx) {
    Drone* agent = &env->agents[idx];
    float x = (float)(idx % 8);
    float y = (float)(idx / 8);
    x = 2.0f*x - 7;
    y = 5 - 1.5f*y;
    agent->target_pos = (Vec3){0.0f, x, y};
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_race(DroneSwarm* env, int idx) {
    Drone* agent = &env->agents[idx];
    agent->target_pos = env->ring_buffer[agent->ring_idx].pos;
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target(DroneSwarm* env, int idx) {
    if (env->task == TASK_IDLE) {
        set_target_idle(env, idx);
    } else if (env->task == TASK_HOVER) {
        set_target_hover(env, idx);
    } else if (env->task == TASK_ORBIT) {
        set_target_orbit(env, idx);
    } else if (env->task == TASK_FOLLOW) {
        set_target_follow(env, idx);
    } else if (env->task == TASK_CUBE) {
        set_target_cube(env, idx);
    } else if (env->task == TASK_CONGO) {
        set_target_congo(env, idx);
    } else if (env->task == TASK_FLAG) {
        set_target_flag(env, idx);
    } else if (env->task == TASK_RACE) {
        set_target_race(env, idx);
    }
}

float compute_reward(DroneSwarm* env, Drone *agent, bool collision) {
    // Distance reward
    float dx = (agent->pos.x - agent->target_pos.x);
    float dy = (agent->pos.y - agent->target_pos.y);
    float dz = (agent->pos.z - agent->target_pos.z);
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    float dist_reward = 1.0 - dist/MAX_DIST;
    //dist = clampf(dist, 0.0f, 1.0f);
    //float dist_reward = 1.0f - dist;

    // Density penalty
    float density_reward = 0.0f;
    if (collision && env->num_agents > 1) {
        Drone *nearest = nearest_drone(env, agent);
        dx = agent->pos.x - nearest->pos.x;
        dy = agent->pos.y - nearest->pos.y;
        dz = agent->pos.z - nearest->pos.z;
        float min_dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (min_dist < 1.0f) {
            density_reward = -1.0f;
            agent->collisions += 1.0f;
        }
    }

    float abs_reward = dist_reward + density_reward;

    agent->last_collision_reward = density_reward;
    agent->last_target_reward = dist_reward;
    agent->last_abs_reward = abs_reward;

    agent->episode_length++;
    agent->score += abs_reward;

    return abs_reward;
}

void reset_agent(DroneSwarm* env, Drone *agent, int idx) {
    agent->uid = uid_tracker++;
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->pos = (Vec3){rndf(-9, 9), rndf(-9, 9), rndf(-9, 9)};
    agent->spawn_pos = agent->pos;
    agent->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->omega = (Vec3){0.0f, 0.0f, 0.0f};
    agent->quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    agent->ring_idx = 0;

    //float size = 0.2f;
    //init_drone(agent, size, 0.0f);
    float size = rndf(0.1f, 0.4);
    init_drone(agent, size, 0.1f);
    compute_reward(env, agent, env->task != TASK_RACE);
}

void c_reset(DroneSwarm *env) {
    env->tick = 0;
    //env->task = rand() % (TASK_N - 1);
    //env->task = TASK_FLAG;
    env->task = TASK_CONGO;
    //env->task = rand() % (TASK_N - 1);
    /*
    if (rand() % 4) {
        env->task = TASK_RACE;
    } else {
        env->task = rand() % (TASK_N - 1);
    }
    */
    //env->task = TASK_RACE;
    //env->task = TASK_HOVER;
    //env->task = TASK_FLAG;

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(env, i);
    }

    for (int i = 0; i < env->max_rings; i++) {
        Ring *ring = &env->ring_buffer[i];
        *ring = (Ring){0};
    }
    if (env->task == TASK_RACE) {
        float ring_radius = 2.0f;
        if (env->max_rings + 1 > 0) {
            env->ring_buffer[0] = rndring(ring_radius);
        }

        for (int i = 1; i < env->max_rings; i++) {
            do {
                env->ring_buffer[i] = rndring(ring_radius);
            } while (norm3(sub3(env->ring_buffer[i].pos, env->ring_buffer[i - 1].pos)) < 2.0f*ring_radius);
        }

        // start drone at least MARGIN away from the first ring
        for (int i = 0; i < env->num_agents; i++) {
            Drone *drone = &env->agents[i];
            do {
                drone->pos = (Vec3){rndf(-9, 9), rndf(-9, 9), rndf(-9, 9)};
            } while (norm3(sub3(drone->pos, env->ring_buffer[0].pos)) < 2.0f*ring_radius);
        }
    }
 
    compute_observations(env);
}

// maps drone unique identifier to env array identifier
int drone_index_lookup(DroneSwarm *env, int uid) {
    for (int aid = 0; aid < env->num_agents; aid++) {
        if (env->agents[aid].uid == uid) {
            return aid;
        }
    }
    return -1;
}



bool check_collision(Rocket* rocket, Drone* drone) {
    float dx = drone->pos.x - rocket->pos.x;
    float dy = drone->pos.y - rocket->pos.y;
    float dz = drone->pos.z - rocket->pos.z;
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);
    return dist < 1.0f;
}


void update_rockets(DroneSwarm *env) {

    for (int r = 0; r < env->rocket_count; r++) {  // increment vel + oob check
        Rocket *rocket = &env->rockets[r];
        rocket->pos = add3(rocket->pos, scalmul3(rocket->vel, DT));
        
        if (rocket->pos.x < -GRID_X || rocket->pos.x > GRID_X ||
            rocket->pos.y < -GRID_Y || rocket->pos.y > GRID_Y ||
            rocket->pos.z < -GRID_Z || rocket->pos.z > GRID_Z) {
            env->rockets[r] = env->rockets[--env->rocket_count]; r--; continue;
        }
    
        for (int d = 0; d < env->num_agents; d++) {    // collision check
            Drone *drone = &env->agents[d];
            if (drone->uid == rocket->parent->uid) continue;
            if (check_collision(rocket, drone)) {
                int shooter_idx = drone_index_lookup(env, rocket->parent->uid);
                int target_idx = d;
                env->rewards[shooter_idx] += HIT_REWARD;
                env->log.rocket_hits += 1.0f;
                env->rewards[target_idx] -= HIT_PUNISH;
                drone->health -= ROCKET_DMG;
                if (drone->health <= 0) {
                    env->rewards[target_idx] -= DEATH_PUNISH;
                    env->terminals[target_idx] = 1;
                }
                env->rockets[r] = env->rockets[env->rocket_count - 1];
                env->rocket_count--;
            }

        }
    }
}










void fire_rocket(Drone* drone, float* actions) {


    // TODO: Implement rocket firing
}
void c_step(DroneSwarm *env) {
    env->tick = (env->tick + 1) % HORIZON;
    update_rockets(env);
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;

        float* atn = &env->actions[7*i];
        move_drone(agent, atn);

        // Check cooldown before firing rocket
        if(atn[4]>0.f && env->rocket_count<MAX_ROCKETS && agent->rocket_cooldown <= 0){
            float az=atn[5]*M_PI;
            float el=atn[6]*1.57079632679f;
            Vec3 dir={cosf(el)*cosf(az),cosf(el)*sinf(az),sinf(el)};
            Rocket r={agent->pos,scalmul3(dir,ROCKET_VEL),agent};
            env->rockets[env->rocket_count++]=r;
            agent->rocket_cooldown = ROCKET_COOLDOWN;
        }

        // check out of bounds
        bool out_of_bounds = agent->pos.x < -GRID_X || agent->pos.x > GRID_X ||
                             agent->pos.y < -GRID_Y || agent->pos.y > GRID_Y ||
                             agent->pos.z < -GRID_Z || agent->pos.z > GRID_Z;

        move_target(env, agent);

        float reward = 0.0f;
        if (env->task == TASK_RACE) {
            Ring *ring = &env->ring_buffer[agent->ring_idx];
            reward = compute_reward(env, agent, true);
            float passed_ring = check_ring(agent, ring);
            if (passed_ring > 0) {
                agent->ring_idx = (agent->ring_idx + 1) % env->max_rings;
                env->log.rings_passed += 1.0f;
                set_target(env, i);
                compute_reward(env, agent, true);
            }
            reward += passed_ring;
        } else {
            // Delta reward
            reward = compute_reward(env, agent, true);
        }

        env->rewards[i] += reward;
        agent->episode_return += reward;

        if (out_of_bounds) {
            env->rewards[i] -= 1;
            env->terminals[i] = 1;
            add_log(env, i, true);
            reset_agent(env, agent, i);
        } else if (env->tick >= HORIZON - 1) {
            env->terminals[i] = 1;
            add_log(env, i, false);
        }
    }
    // Decrease cooldown for all drones
    for (int i = 0; i < env->num_agents; i++) {
        if (env->agents[i].rocket_cooldown > 0) {
            env->agents[i].rocket_cooldown--;
        }
    }

    if (env->tick >= HORIZON - 1) {
        c_reset(env);
    }

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(DroneSwarm *env) {
    if (env->client != NULL) {
        c_close_client(env->client);
    }
}

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = 0.01f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 7.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 90.0f);
        update_camera_position(client);
    }
}

Client *make_client(DroneSwarm *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    InitWindow(WIDTH, HEIGHT, "PufferLib DroneSwarm");

#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->camera_distance = 40.0f;
    client->camera_azimuth = 0.0f;
    client->camera_elevation = PI / 10.0f;
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0.0f, 0.0f};

    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    // Initialize trail buffer
    client->trails = (Trail*)calloc(env->num_agents, sizeof(Trail));
    for (int i = 0; i < env->num_agents; i++) {
        Trail* trail = &client->trails[i];
        trail->index = 0;
        trail->count = 0;
        for (int j = 0; j < TRAIL_LENGTH; j++) {
            trail->pos[j] = env->agents[i].pos;
        }
    }

    return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void DrawRing3D(Ring ring, float thickness, Color entryColor, Color exitColor) {
    float half_thick = thickness / 2.0f;

    Vector3 center_pos = {ring.pos.x, ring.pos.y, ring.pos.z};

    Vector3 entry_start_pos = {center_pos.x - half_thick * ring.normal.x,
                               center_pos.y - half_thick * ring.normal.y,
                               center_pos.z - half_thick * ring.normal.z};

    DrawCylinderWiresEx(entry_start_pos, center_pos, ring.radius, ring.radius, 32, entryColor);

    Vector3 exit_end_pos = {center_pos.x + half_thick * ring.normal.x,
                            center_pos.y + half_thick * ring.normal.y,
                            center_pos.z + half_thick * ring.normal.z};

    DrawCylinderWiresEx(center_pos, exit_end_pos, ring.radius, ring.radius, 32, exitColor);
}





void draw_rockets(DroneSwarm *env) {
    for(int i = 0; i < env->rocket_count; i++) {
        Rocket *rocket = &env->rockets[i];
        DrawSphere((Vector3){rocket->pos.x, rocket->pos.y, rocket->pos.z}, 0.1f, RED);
    }
}

void c_render(DroneSwarm *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }

    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    if (IsKeyPressed(KEY_SPACE)) {
        player_active = !player_active;
        player_idx = player_active ? env->num_agents - 1 : -1;
    }

    if (!player_active) {
        handle_camera_controls(env->client);
    }
    Client *client = env->client;
    if (player_active) {
        Drone *p = &env->agents[player_idx];
        Vec3 fwd = {cosf(player_pitch)*cosf(player_yaw), cosf(player_pitch)*sinf(player_yaw), sinf(player_pitch)};
        client->camera.position = (Vector3){p->pos.x, p->pos.y, p->pos.z};
        client->camera.target = (Vector3){p->pos.x + fwd.x, p->pos.y + fwd.y, p->pos.z + fwd.z};
        client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    } else {
        Client *client = env->client;
    }

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        Trail *trail = &client->trails[i];
        trail->pos[trail->index] = agent->pos;
        trail->index = (trail->index + 1) % TRAIL_LENGTH;
        if (trail->count < TRAIL_LENGTH) {
            trail->count++;
        }
        if (env->terminals[i]) {
            trail->index = 0;
            trail->count = 0;
        }
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    BeginMode3D(client->camera);

    // draws bounding cube
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_X * 2.0f,
        GRID_Y * 2.0f, GRID_Z * 2.0f, WHITE);
    draw_rockets(env);
    for (int i = 0; i < env->num_agents; i++) {
        if (player_active && i == player_idx) continue;
        Drone *agent = &env->agents[i];

        // draws drone body
        Color body_color = FLAG_COLORS[i];
        DrawSphere((Vector3){agent->pos.x, agent->pos.y, agent->pos.z}, 0.3f, body_color);

        // draws rotors according to thrust
        float T[4];
        for (int j = 0; j < 4; j++) {
            float rpm = (env->actions[4*i + j] + 1.0f) * 0.5f * agent->max_rpm;
            T[j] = agent->k_thrust * rpm * rpm;
        }

        const float rotor_radius = 0.15f;
        const float visual_arm_len = agent->arm_len * 4.0f;

        Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                      {-visual_arm_len, 0.0f, 0.0f},
                                      {0.0f, +visual_arm_len, 0.0f},
                                      {0.0f, -visual_arm_len, 0.0f}};

        //Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};
        Color base_colors[4] = {body_color, body_color, body_color, body_color};

        for (int j = 0; j < 4; j++) {
            Vec3 world_off = quat_rotate(agent->quat, rotor_offsets_body[j]);

            Vector3 rotor_pos = {agent->pos.x + world_off.x, agent->pos.y + world_off.y,
                                 agent->pos.z + world_off.z};

            float rpm = (env->actions[4*i + j] + 1.0f) * 0.5f * agent->max_rpm;
            float intensity = 0.75f + 0.25f * (rpm / agent->max_rpm);
            Color rotor_color = (Color){(unsigned char)fminf(base_colors[j].r * intensity, 255.0f),
                                        (unsigned char)fminf(base_colors[j].g * intensity, 255.0f),
                                        (unsigned char)fminf(base_colors[j].b * intensity, 255.0f), 255};

            DrawSphere(rotor_pos, rotor_radius, rotor_color);

            DrawCylinderEx((Vector3){agent->pos.x, agent->pos.y, agent->pos.z}, rotor_pos, 0.02f, 0.02f, 8,
                           BLACK);
        }

        // draws line with direction and magnitude of velocity / 10
        if (norm3(agent->vel) > 0.1f) {
            DrawLine3D((Vector3){agent->pos.x, agent->pos.y, agent->pos.z},
                       (Vector3){agent->pos.x + agent->vel.x * 0.1f, agent->pos.y + agent->vel.y * 0.1f,
                                 agent->pos.z + agent->vel.z * 0.1f},
                       MAGENTA);
        }

        // Draw trailing path
        Trail *trail = &client->trails[i];
        if (trail->count <= 2) {
            continue;
        }
        for (int j = 0; j < trail->count - 1; j++) {
            int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
            float alpha = (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f; // fade out
            Color trail_color = ColorAlpha((Color){0, 187, 187, 255}, alpha);
            DrawLine3D((Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
                       (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
                       trail_color);
        }

    }

    // Rings
    if (env->task == TASK_RACE) {
        float ring_thickness = 0.2f;
        for (int i = 0; i < env->max_rings; i++) {
            Ring ring = env->ring_buffer[i];
            DrawRing3D(ring, ring_thickness, GREEN, BLUE);
        }
    }

    if (IsKeyDown(KEY_TAB)) {
        for (int i = 0; i < env->num_agents; i++) {
            Drone *agent = &env->agents[i];
            Vec3 target_pos = agent->target_pos;
            DrawSphere((Vector3){target_pos.x, target_pos.y, target_pos.z}, 0.45f, (Color){0, 255, 255, 100});
        }
    }

    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);
    DrawText(TextFormat("Task: %s", TASK_NAMES[env->task]), 10, 50, 16, PUFF_WHITE);

    EndDrawing();
}


