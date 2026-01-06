/* Battle: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>
#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"
#include "simplex.h"

#define RLIGHTS_IMPLEMENTATION
#include "rlights.h"


#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION 330
#else
    #define GLSL_VERSION 100
#endif

#define MAX_SPEED 0.01f
#define MAX_FACTORY_SPEED 0.001f

#define DRONE 0
#define MOTHERSHIP 1
#define FIGHTER 2
#define BOMBER 3
#define INFANTRY 4
#define TANK 5
#define ARTILLERY 6
#define BASE 7

#define AGENT_OBS 16

static inline float clampf(float v, float min, float max) {
  if (v < min)
    return min;
  if (v > max)
    return max;
  return v;
}

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

float clip_angle(float theta) {
    if (theta < -PI) {
        return theta + 2.0f*PI;
    } else if (theta > PI) {
        return theta - 2.0f*PI;
    }
    return theta;
}

float randf(float min, float max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

float randi(int min, int max) {
    return min + (max - min)*(float)rand()/(float)RAND_MAX;
}

typedef struct {
    float perf;
    float score;
    float collision_rate;
    float oob_rate;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    Camera3D camera;
    Light light;
    Model models[8];
    Mesh* mesh;
    Model model;
    Shader light_shader;
    Shader terrain_shader;
    Texture2D terrain_texture;
    Texture2D vehicle_texture;
    int terrain_shader_loc;
    unsigned char *terrain_data;
} Client;

typedef struct {
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float speed;
    float health;
    float max_turn;
    float max_speed;
    float attack_damage;
    float attack_range;
    Quaternion orientation;
    int army;
    int unit;
    int target;
    int episode_length;
    float episode_return;
} Entity;

typedef struct {
    Log log;
    Client* client;
    Entity* agents;
    Entity* bases;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    int width;
    int height;
    float size_x;
    float size_y;
    float size_z;
    int terrain_width;
    int terrain_height;
    int num_agents;
    int num_armies;
    float* terrain;
} Battle;

int map_idx(Battle* env, float x, float y) {
    return env->terrain_width*(int)y + (int)x;
}

float ground_height(Battle* env, float x, float z) {
    int agent_map_x = 128*x + 128*env->size_x;
    int agent_map_z = 128*z + 128*env->size_z;
    if (agent_map_x == 256*env->size_x) {
        agent_map_x -= 1;
    }
    if (agent_map_z == 256*env->size_z) {
        agent_map_z -= 1;
    }
    int idx = map_idx(env, agent_map_x, agent_map_z);
    float terrain_height = env->terrain[idx];
    return (terrain_height - 128.0f*env->size_y) / 128.0f;
}

void perlin_noise(float* map, int width, int height,
        float base_frequency, int octaves, int offset_x, int offset_y, float glob_scale) {
    float frequencies[octaves];
    for (int i = 0; i < octaves; i++) {
        frequencies[i] = base_frequency*pow(2, i);
    }

    float min_value = FLT_MAX;
    float max_value = FLT_MIN;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            for (int oct = 0; oct < octaves; oct++) {
                float freq = frequencies[oct];
                map[adr] += (1.0/pow(2, oct))*noise2(freq*c + offset_x, freq*r + offset_y);
            }
            float val = map[adr];
            if (val < min_value) {
                min_value = val;
            }
            if (val > max_value) {
                max_value = val;
            }
        }
    }

    float scale = 1.0/(max_value - min_value);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*width + c;
            map[adr] = glob_scale * scale * (map[adr] - min_value);
            if (map[adr] < 16.0f) {
                map[adr] = 0.0f;
            } else {
                map[adr] -= 16.0f;
            }
        }
    }
}

void init(Battle* env) {
    env->agents = calloc(env->num_agents, sizeof(Entity));
    env->bases = calloc(env->num_armies, sizeof(Entity));
    env->terrain_width = 256*env->size_x;
    env->terrain_height = 256*env->size_z;
    env->terrain = calloc(env->terrain_width*env->terrain_height, sizeof(float));
    perlin_noise(env->terrain, env->terrain_width, env->terrain_height, 1.0/2048.0, 8, 0, 0, 256);
}

void update_abilities(Entity* agent) {
    if (agent->unit == DRONE) {
        agent->health = 0.4f;
        agent->attack_damage = 0.1f;
        agent->attack_range = 0.15f;
        agent->max_turn = 2.0f;
        agent->max_speed = 1.0f;
    } else if (agent->unit == FIGHTER) {
        agent->health = 1.0f;
        agent->attack_damage = 0.5f;
        agent->attack_range = 0.25f;
        agent->max_turn = 1.0f;
        agent->max_speed = 0.75f;
    } else if (agent->unit == MOTHERSHIP) {
        agent->health = 10.0f;
        agent->attack_damage = 2.0f;
        agent->attack_range = 0.4f;
        agent->max_turn = 0.5f;
        agent->max_speed = 0.5f;
    } else if (agent->unit == BOMBER) {
        agent->health = 1.0f;
        agent->attack_damage = 1.0f;
        agent->attack_range = 0.1f;
        agent->max_turn = 0.5f;
        agent->max_speed = 0.5f;
    } else if (agent->unit == INFANTRY) {
        agent->health = 0.2f;
        agent->attack_damage = 0.2f;
        agent->attack_range = 0.2f;
        agent->max_turn = 2.0f;
        agent->max_speed = 0.25f;
    } else if (agent->unit == TANK) {
        agent->health = 2.0f;
        agent->attack_damage = 0.5f;
        agent->attack_range = 0.25f;
        agent->max_turn = 0.25f;
        agent->max_speed = 0.75f;
    } else if (agent->unit == ARTILLERY) {
        agent->health = 2.0f;
        agent->attack_damage = 2.0f;
        agent->attack_range = 0.7f;
        agent->max_turn = 0.5f;
        agent->max_speed = 0.25f;
    }
}

void respawn(Battle* env, int idx) {
    Entity* agent = &env->agents[idx];
    int army = agent->army;
    agent->orientation = QuaternionIdentity();
    agent->vx = 0;
    agent->vy = 0;
    agent->vz = 0;

    if (agent->unit == DRONE) {
        int team_mothership_idx = 64*(idx / 64); // Hardcoded per army
        agent->x = env->agents[team_mothership_idx].x;
        agent->y = env->agents[team_mothership_idx].y;
        agent->z = env->agents[team_mothership_idx].z;
        if (agent->unit == INFANTRY || agent->unit == TANK || agent->unit == ARTILLERY) {
            agent->y = ground_height(env, agent->x, agent->z);
        }
        return;
    }

    Entity* base = &env->bases[army];
    agent->x = base->x;
    agent->z = base->z;
    float height = ground_height(env, agent->x, agent->z);
    if (agent->unit == INFANTRY || agent->unit == TANK || agent->unit == ARTILLERY) {
        agent->y = height;
    } else {
        agent->y = clampf(height + 0.2f, -env->size_y, env->size_y);
    }

    return;
}


bool attack_air(Entity *agent, Entity *target) {
    float dx = target->x - agent->x;
    float dy = target->y - agent->y;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dy*dy + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    Vector3 forward = Vector3RotateByQuaternion((Vector3){0, 0, 1}, agent->orientation);
    forward = Vector3Normalize(forward);

    // Unit vec to target
    Vector3 to_target = {dx, dy, dz};
    to_target = Vector3Normalize(to_target);

    float angle = Vector3Angle(forward, to_target);
    if (angle < PI/6.0f) {
        return true;
    }
    return false;
}

bool attack_ground(Entity *agent, Entity *target) {
    if (target->unit == FIGHTER) {
        return false;
    }
    if (target->unit == MOTHERSHIP) {
        return false;
    }
    if (target->unit == BOMBER) {
        return false;
    }
    if (target->unit == DRONE) {
        return false;
    }

    float dx = target->x - agent->x;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    Vector3 forward = Vector3RotateByQuaternion((Vector3){0, 0, 1}, agent->orientation);
    forward = Vector3Normalize(forward);

    // Unit vec to target
    Vector3 to_target = {dx, 0, dz};
    to_target = Vector3Normalize(to_target);

    float angle = Vector3Angle(forward, to_target);
    if (angle < PI/6) {
        return true;
    }
    return false;
}

bool attack_bomber(Entity *agent, Entity *target) {
    if (target->unit == DRONE) {
        return false;
    }
    if (target->unit == FIGHTER) {
        return false;
    }
    if (target->unit == MOTHERSHIP) {
        return false;
    }
    if (target->unit == BOMBER) {
        return false;
    }

    float dx = target->x - agent->x;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    return true;
}

bool attack_aa(Entity *agent, Entity *target) {
    if (target->unit == INFANTRY) {
        return false;
    }
    if (target->unit == TANK) {
        return false;
    }
    if (target->unit == ARTILLERY) {
        return false;
    }

    float dx = target->x - agent->x;
    float dy = target->y - agent->y;
    float dz = target->z - agent->z;
    float dd = sqrtf(dx*dx + dz*dz);

    if (dd > agent->attack_range) {
        return false;
    }

    Vector3 forward = Vector3RotateByQuaternion((Vector3){0, 0, 1}, agent->orientation);
    forward = Vector3Normalize(forward);

    // Unit vec to target
    Vector3 to_target = {dx, dy, dz};
    to_target = Vector3Normalize(to_target);

    float angle = Vector3Angle(forward, to_target);
    if (angle < PI/6) {
        return true;
    }
}

void move_basic(Battle* env, Entity* agent, float* actions) {
    float d_vx = actions[0]/100.0f;
    float d_vy = actions[1]/100.0f;
    float d_vz = actions[2]/100.0f;

    agent->vx += d_vx;
    agent->vy += d_vy;
    agent->vz += d_vz;

    agent->vx = clip(agent->vx, -MAX_SPEED, MAX_SPEED);
    agent->vy = clip(agent->vy, -MAX_SPEED, MAX_SPEED);
    agent->vz = clip(agent->vz, -MAX_SPEED, MAX_SPEED);

    agent->x += agent->vx;
    agent->y += agent->vy;
    agent->z += agent->vz;

    agent->x = clip(agent->x, -env->size_x, env->size_x);
    agent->y = clip(agent->y, -env->size_y, env->size_y);
    agent->z = clip(agent->z, -env->size_z, env->size_z);
}

void move_ground(Battle* env, Entity* agent, float* actions) {
    float d_theta = -actions[1]/10.0f;

    // Update speed and clamp
    agent->speed = agent->max_speed * MAX_SPEED;

    Quaternion q_y = QuaternionFromAxisAngle((Vector3){0, 1, 0}, d_theta);
    agent->orientation = QuaternionMultiply(q_y, agent->orientation);

    Vector3 forward = Vector3RotateByQuaternion((Vector3){0, 0, 1}, agent->orientation);
    forward = Vector3Normalize(forward);

    agent->speed = agent->max_speed * MAX_SPEED;
    agent->vx = agent->speed * forward.x;
    agent->vz = agent->speed * forward.z;
    agent->x += agent->vx;
    agent->z += agent->vz;

    agent->x = clip(agent->x, -env->size_x, env->size_x);
    agent->z = clip(agent->z, -env->size_z, env->size_z);
    agent->y = ground_height(env, agent->x, agent->z);
}

Entity* nearest_enemy(Battle* env, Entity* agent) {
    Entity* nearest = NULL;
    float nearest_dist = 999999;
    for (int i=0; i<env->num_agents; i++) {
        Entity* other = &env->agents[i];
        if (other->army == agent->army) {
            continue;
        }
        float dx = other->x - agent->x;
        float dy = other->y - agent->y;
        float dz = other->z - agent->z;
        float dd = dx*dx + dy*dy + dz*dz;
        if (dd < nearest_dist) {
            nearest_dist = dd;
            nearest = other;
        }
    }
    return nearest;
}

// Cheats physics and moves directly to the nearest enemy
void scripted_move(Battle* env, Entity* agent, bool is_air) {
    Entity* target = nearest_enemy(env, agent);
    if (target == NULL) {
        return;
    }
    float dx = target->x - agent->x;
    float dy = target->y - agent->y;
    float dz = target->z - agent->z;

    // Add some noise
    dx += randf(-0.1f, 0.1f);
    dy += randf(-0.1f, 0.1f);
    dz += randf(-0.1f, 0.1f);

    float dd = dx*dx + dz*dz;
    if (is_air) {
        dd += dy*dy;
    }

    dd = sqrtf(dd);
    dx /= dd;
    dy /= dd;
    dz /= dd;

    
    float target_x;
    float target_y;
    float target_z;
    if (dd > 0.05f) {
        target_x = agent->x + dx*agent->max_speed*MAX_SPEED;
        target_y = agent->y + dy*agent->max_speed*MAX_SPEED;
        target_z = agent->z + dz*agent->max_speed*MAX_SPEED;
    } else {
        target_x = agent->x - dx*agent->max_speed*MAX_SPEED;
        target_y = agent->y - dy*agent->max_speed*MAX_SPEED;
        target_z = agent->z - dz*agent->max_speed*MAX_SPEED;
    }
    
    float height = ground_height(env, target_x, target_z);
    if (is_air) {
        if (target_y < height + 0.5f) {
            target_y = height + 0.5f;
        }
    } else {
        target_y = height;
    }

    agent->x = target_x;
    agent->y = target_y;
    agent->z = target_z;

    agent->x = clip(agent->x, -env->size_x, env->size_x);
    agent->y = clip(agent->y, -env->size_y, env->size_y);
    agent->z = clip(agent->z, -env->size_z, env->size_z);

    // Update orientation to target
    Vector3 target_forward = {dx, 0, dz};
    if (is_air) {
        target_forward.y = dy;
    }
    target_forward = Vector3Normalize(target_forward);

    Vector3 current_forward = Vector3RotateByQuaternion((Vector3){0, 0, 1}, agent->orientation);
    current_forward = Vector3Normalize(current_forward);

    Quaternion q = QuaternionFromVector3ToVector3(current_forward, target_forward);
    agent->orientation = QuaternionMultiply(q, agent->orientation);
}

void move_ship(Battle* env, Entity* agent, float* actions, int i) {
    // Compute deltas from actions (same as original)
    float d_pitch = agent->max_turn * actions[0] / 10.0f;
    float d_roll = agent->max_turn * actions[1] / 10.0f;

    // Update speed and clamp
    agent->speed = agent->max_speed * MAX_SPEED;

    Vector3 forward = Vector3RotateByQuaternion((Vector3){0, 0, 1}, agent->orientation);
    forward = Vector3Normalize(forward);

    Vector3 local_up = Vector3RotateByQuaternion((Vector3){0, 1, 0}, agent->orientation);
    local_up = Vector3Normalize(local_up);

    Vector3 right = Vector3CrossProduct(forward, local_up); // Ship's local right
    right = Vector3Normalize(right);

    // Create rotation quaternions
    /*
    if (i == 0) {
        printf("actions: %d %d %d\n", actions[0], actions[1], actions[2]);
        printf("orientation: %f %f %f %f\n", agent->orientation.w, agent->orientation.x, agent->orientation.y, agent->orientation.z);
        printf("Local up: %f %f %f\n", local_up.x, local_up.y, local_up.z);
        printf("Forward: %f %f %f\n", forward.x, forward.y, forward.z);
        printf("Right: %f %f %f\n", right.x, right.y, right.z);
        printf("d_pitch: %f\n, d_roll: %f\n", d_pitch, d_roll);
    }
    */

    float d_yaw = 0.0;
    Quaternion q_yaw = QuaternionFromAxisAngle(local_up, d_yaw);
    Quaternion q_roll = QuaternionFromAxisAngle(forward, d_roll);
    Quaternion q_pitch = QuaternionFromAxisAngle(right, d_pitch);

    /*
    if (i == 0) {
        printf("q_yaw: %f %f %f %f\n", q_yaw.w, q_yaw.x, q_yaw.y, q_yaw.z);
        printf("q_roll: %f %f %f %f\n", q_roll.w, q_roll.x, q_roll.y, q_roll.z);
        printf("q_pitch: %f %f %f %f\n", q_pitch.w, q_pitch.x, q_pitch.y, q_pitch.z);
    }
    */

    Quaternion q = QuaternionMultiply(q_roll, QuaternionMultiply(q_pitch, q_yaw));
    q = QuaternionNormalize(q);

    forward = Vector3RotateByQuaternion(forward, q);
    forward = Vector3Normalize(forward);

    agent->orientation = QuaternionMultiply(q, agent->orientation);

    // Jank plane physics
    Vector3 v = {
        agent->speed * (forward.x + local_up.x),
        agent->speed * (forward.y + local_up.y - 1.0f),
        agent->speed * (forward.z + local_up.z)
    };

    agent->x += v.x;
    agent->y += v.y;
    agent->z += v.z;

    // Just for visualization
    agent->vx = v.x;
    agent->vy = v.y;
    agent->vz = v.z;

    // Clamp position to environment bounds
    agent->x = clampf(agent->x, -env->size_x, env->size_x);
    agent->y = clampf(agent->y, -env->size_y, env->size_y);
    agent->z = clampf(agent->z, -env->size_z, env->size_z);
}

typedef struct {
    float distance;
    float dx;
    float dy;
    float dz;
    float same_team;
    int idx;
} AgentObs;

int compare_agent_obs(const void* a, const void* b) {
    AgentObs* oa = (AgentObs*)a;
    AgentObs* ob = (AgentObs*)b;
    if (oa->distance < ob->distance) {
        return -1;
    } else if (oa->distance > ob->distance) {
        return 1;
    }
    return 0;
}

void compute_observations(Battle* env) {
    AgentObs agent_obs[env->num_agents];

    int obs_idx = 0;
    for (int a=0; a<env->num_agents/2; a++) {
        assert(obs_idx == a*(6*env->num_armies + 19 + 8));

        // Distance to each base
        Entity* agent = &env->agents[a];
        int team = agent->army;
        float dists[env->num_armies];
        for (int i=0; i<env->num_armies; i++) {
            dists[i] = 999999;
        }
        for (int f=0; f<env->num_armies; f++) {
            Entity* base = &env->bases[f];
            float dx = base->x - agent->x;
            float dy = base->y - agent->y;
            float dz = base->z - agent->z;
            float dd = dx*dx + dy*dy + dz*dz;
            int type = f % env->num_armies;
            if (dd < dists[type]) {
                dists[type] = dd;
                env->observations[obs_idx + 3*type] = dx;
                env->observations[obs_idx + 3*type + 1] = dy;
                env->observations[obs_idx + 3*type + 2] = dz;
            }
        }
        obs_idx += 3*env->num_armies;


        // Distance to each agent. Slow O(n^2) naive implementation
        float x = agent->x;
        float y = agent->y;
        float z = agent->z;
        for (int i=0; i<env->num_agents; i++) {
            Entity* other = &env->agents[i];
            float dx = other->x - x;
            float dy = other->y - y;
            float dz = other->z - z;
            float distance = dx*dx + dy*dy + dz*dz;
            AgentObs* o = &agent_obs[i];
            o->dx = dx;
            o->dy = dy;
            o->dz = dz;
            if (other->army == agent->army) {
                o->same_team = 1.0f;
                o->distance = 99999.0f;
            } else {
                o->same_team = 0.0f;
                o->distance = distance;
            }
            o->idx = i;
        }
        qsort(agent_obs, env->num_agents, sizeof(AgentObs), compare_agent_obs);

        for (int i=0; i<AGENT_OBS; i++) {
            env->observations[obs_idx++] = agent_obs[i].dx;
            env->observations[obs_idx++] = agent_obs[i].dy;
            env->observations[obs_idx++] = agent_obs[i].dz;
            env->observations[obs_idx++] = agent_obs[i].same_team;
        }

        // Individual agent stats
        env->observations[obs_idx++] = agent->vx/MAX_SPEED;
        env->observations[obs_idx++] = agent->vy/MAX_SPEED;
        env->observations[obs_idx++] = agent->vz/MAX_SPEED;
        env->observations[obs_idx++] = agent->orientation.w;
        env->observations[obs_idx++] = agent->orientation.x;
        env->observations[obs_idx++] = agent->orientation.y;
        env->observations[obs_idx++] = agent->orientation.z;
        env->observations[obs_idx++] = agent->x;
        env->observations[obs_idx++] = agent->y;
        env->observations[obs_idx++] = agent->z;
        env->observations[obs_idx++] = agent->y - ground_height(env, agent->x, agent->z);
        env->observations[obs_idx++] = abs(agent->x) - 0.95f*env->size_x;
        env->observations[obs_idx++] = abs(agent->z) - 0.95f*env->size_z;
        env->observations[obs_idx++] = abs(agent->y) - 0.95f*env->size_y;
        env->observations[obs_idx++] = agent->speed;
        env->observations[obs_idx++] = agent->health;
        env->observations[obs_idx++] = agent->max_turn;
        env->observations[obs_idx++] = agent->max_speed;
        env->observations[obs_idx++] = agent->attack_damage;
        env->observations[obs_idx++] = agent->attack_range;
        env->observations[obs_idx++] = env->rewards[a];
        env->observations[obs_idx++] = env->terminals[a];

        // Hardcoded 8 unit types
        memset(&env->observations[obs_idx], 0, 8*sizeof(float));
        env->observations[obs_idx + agent->unit] = 1.0f;
        obs_idx += 8;
    }
}

// Required function
void c_reset(Battle* env) {
    int agents_per_army = env->num_agents / env->num_armies;
    for (int i=0; i<env->num_armies; i++) {
        bool spawn = false;
        Entity* base = &env->bases[i];
        while (!spawn) {
            base->x = randf(0.5 - env->size_x, env->size_x - 0.5);
            base->z = randf(0.5 - env->size_z, env->size_z - 0.5);
            base->y = ground_height(env, base->x, base->z);
            base->army = i;
            spawn = true;

            for (int j=0; j<i; j++) {
                Entity* other = &env->bases[j];
                float dx = other->x - base->x;
                float dz = other->z - base->z;
                float dd = sqrtf(dx*dx + dz*dz);
                if (dd < 2.0f) {
                    spawn = false;
                    break;
                }
            }
        }
    }

    for (int army=0; army<env->num_armies; army++) {
        for (int i=0; i<agents_per_army; i++) {
            int idx = army*agents_per_army + i;
            Entity* agent = &env->agents[idx];
            if (i % 64 == 0) {
                agent->unit = MOTHERSHIP;
            } else if (i % 64 <= 4) {
                agent->unit = TANK;
            } else if (i % 64 <= 6) {
                agent->unit = ARTILLERY;
            } else if (i % 64 <= 10) {
                agent->unit = BOMBER;
            } else if (i % 64 <= 14) {
                agent->unit = FIGHTER;
            } else if (i % 64 <= 32) {
                agent->unit = INFANTRY;
            } else {
                agent->unit = DRONE;
            }

            agent->army = army;
            agent->orientation = QuaternionIdentity();
            agent->episode_length = 0;
            agent->target = -1;
            update_abilities(agent);
            respawn(env, idx);
        }
    }
    compute_observations(env);
}

void c_step(Battle* env) {
    memset(env->rewards, 0, env->num_agents/2*sizeof(float));
    memset(env->terminals, 0, env->num_agents/2*sizeof(unsigned char));

    for (int i=0; i<env->num_agents; i++) {
        Entity* agent = &env->agents[i];
        agent->episode_length += 1;
        agent->target = -1;

        bool done = false;
        float collision = 0.0f;
        float oob = 0.0f;
        float reward = 0.0f;
        if (agent->health <= 0) {
            done = true;
            reward = 0.0f;
        }
        if (agent->unit == DRONE || agent->unit == FIGHTER || agent->unit == BOMBER || agent->unit == MOTHERSHIP) {
            // Crash into terrain
            float terrain_height = ground_height(env, agent->x, agent->z);
            if (agent->y < terrain_height) {
                collision = 1.0f;
                done = true;
                reward = -1.0f;
            }
        }
        if (
            agent->x < -0.95f*env->size_x || agent->x > 0.95f*env->size_x ||
            agent->z < -0.95f*env->size_z || agent->z > 0.95f*env->size_z ||
            agent->y > 0.95f*env->size_y
        ) {
            done = true;
            reward = -1.0f;
            oob = 1.0f;
        }

        if (done) {
            update_abilities(agent);
            respawn(env, i);
            agent->episode_return += reward;
            if (i < env->num_agents/2) {
                env->rewards[i] = reward;
                env->terminals[i] = 1;
                env->log.score = env->log.episode_return;
                env->log.episode_length += agent->episode_length;
                env->log.episode_return += agent->episode_return;
                env->log.collision_rate += collision;
                env->log.oob_rate += oob;
                env->log.n++;

            }
            agent->episode_length = 0;
            agent->episode_return = 0;
        }

        //move_basic(env, agent, env->actions + 3*i);
        if (agent->unit == INFANTRY || agent->unit == TANK || agent->unit == ARTILLERY) {
            if (i < env->num_agents/2) {
                move_ground(env, agent, env->actions + 3*i);
            } else {
                scripted_move(env, agent, false);
            }
        } else {
            if (i < env->num_agents/2) {
                move_ship(env, agent, env->actions + 3*i, i);
            } else {
                scripted_move(env, agent, true);
            }
        }
    }

    for (int i=0; i<env->num_agents; i++) {
        Entity* agent = &env->agents[i];
        for (int j=0; j<env->num_agents; j++) {
            if (j == i) {
                continue;
            }
            Entity* target = &env->agents[j];
            if (agent->army == target->army) {
                continue;
            }
            bool can_attack = false;
            if (agent->unit == INFANTRY || agent->unit == TANK) {
                can_attack = attack_ground(agent, target);
            } else if (agent->unit == ARTILLERY) {
                can_attack = attack_aa(agent, target);
            } else if (agent->unit == BOMBER) {
                can_attack = attack_bomber(agent, target);
            } else {
                can_attack = attack_air(agent, target);
            }
            if (!can_attack) {
                continue;
            }
            agent->target = j;
            if (i < env->num_agents/2) {
                env->rewards[i] += 0.25f;
                agent->episode_return += 0.25f;
            }
            target->health -= agent->attack_damage;
            break;
        }
    }

    if (rand() % 9000 == 0) {
        c_reset(env);
    }

    compute_observations(env);
}

Color COLORS[8] = {
    (Color){0, 255, 255, 255},
    (Color){255, 0, 0, 255},
    (Color){0, 255, 0, 255},
    (Color){255, 255, 0, 255},
    (Color){255, 0, 255, 255},
    (Color){0, 0, 255, 255},
    (Color){128, 255, 0, 255},
    (Color){255, 128, 0, 255},
};

Mesh* create_heightmap_mesh(float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    // NOTE: One vertex per pixel
    Mesh* mesh = (Mesh*)calloc(1, sizeof(Mesh));
    mesh->triangleCount = (mapX - 1)*(mapZ - 1)*2;    // One quad every four pixels

    mesh->vertexCount = mesh->triangleCount*3;

    mesh->vertices = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->normals = (float *)RL_MALLOC(mesh->vertexCount*3*sizeof(float));
    mesh->texcoords = (float *)RL_MALLOC(mesh->vertexCount*2*sizeof(float));
    mesh->colors = NULL;
    UploadMesh(mesh, false);
    return mesh;
}

void update_heightmap_mesh(Mesh* mesh, float* heightMap, Vector3 size) {
    int mapX = size.x;
    int mapZ = size.z;

    int vCounter = 0;       // Used to count vertices float by float
    int tcCounter = 0;      // Used to count texcoords float by float
    int nCounter = 0;       // Used to count normals float by float

    //Vector3 scaleFactor = { size.x/(mapX - 1), 1.0f, size.z/(mapZ - 1) };
    Vector3 scaleFactor = { 1.0f, 1.0f, 1.0f};

    Vector3 vA = { 0 };
    Vector3 vB = { 0 };
    Vector3 vC = { 0 };
    Vector3 vN = { 0 };

    for (int z = 0; z < mapZ-1; z++)
    {
        for (int x = 0; x < mapX-1; x++)
        {
            // Fill vertices array with data
            //----------------------------------------------------------

            // one triangle - 3 vertex
            mesh->vertices[vCounter] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 1] = heightMap[x + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 2] = (float)z*scaleFactor.z;

            mesh->vertices[vCounter + 3] = (float)x*scaleFactor.x;
            mesh->vertices[vCounter + 4] = heightMap[x + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 5] = (float)(z + 1)*scaleFactor.z;

            mesh->vertices[vCounter + 6] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 7] = heightMap[(x + 1) + z*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 8] = (float)z*scaleFactor.z;

            // Another triangle - 3 vertex
            mesh->vertices[vCounter + 9] = mesh->vertices[vCounter + 6];
            mesh->vertices[vCounter + 10] = mesh->vertices[vCounter + 7];
            mesh->vertices[vCounter + 11] = mesh->vertices[vCounter + 8];

            mesh->vertices[vCounter + 12] = mesh->vertices[vCounter + 3];
            mesh->vertices[vCounter + 13] = mesh->vertices[vCounter + 4];
            mesh->vertices[vCounter + 14] = mesh->vertices[vCounter + 5];

            mesh->vertices[vCounter + 15] = (float)(x + 1)*scaleFactor.x;
            mesh->vertices[vCounter + 16] = heightMap[(x + 1) + (z + 1)*mapX]*scaleFactor.y;
            mesh->vertices[vCounter + 17] = (float)(z + 1)*scaleFactor.z;
            vCounter += 18;     // 6 vertex, 18 floats

            // Fill texcoords array with data
            //--------------------------------------------------------------
            mesh->texcoords[tcCounter] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 1] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 2] = (float)x/(mapX - 1);
            mesh->texcoords[tcCounter + 3] = (float)(z + 1)/(mapZ - 1);

            mesh->texcoords[tcCounter + 4] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 5] = (float)z/(mapZ - 1);

            mesh->texcoords[tcCounter + 6] = mesh->texcoords[tcCounter + 4];
            mesh->texcoords[tcCounter + 7] = mesh->texcoords[tcCounter + 5];

            mesh->texcoords[tcCounter + 8] = mesh->texcoords[tcCounter + 2];
            mesh->texcoords[tcCounter + 9] = mesh->texcoords[tcCounter + 3];

            mesh->texcoords[tcCounter + 10] = (float)(x + 1)/(mapX - 1);
            mesh->texcoords[tcCounter + 11] = (float)(z + 1)/(mapZ - 1);
            tcCounter += 12;    // 6 texcoords, 12 floats

            // Fill normals array with data
            //--------------------------------------------------------------
            for (int i = 0; i < 18; i += 9)
            {
                vA.x = mesh->vertices[nCounter + i];
                vA.y = mesh->vertices[nCounter + i + 1];
                vA.z = mesh->vertices[nCounter + i + 2];

                vB.x = mesh->vertices[nCounter + i + 3];
                vB.y = mesh->vertices[nCounter + i + 4];
                vB.z = mesh->vertices[nCounter + i + 5];

                vC.x = mesh->vertices[nCounter + i + 6];
                vC.y = mesh->vertices[nCounter + i + 7];
                vC.z = mesh->vertices[nCounter + i + 8];

                vN = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(vB, vA), Vector3Subtract(vC, vA)));

                mesh->normals[nCounter + i] = vN.x;
                mesh->normals[nCounter + i + 1] = vN.y;
                mesh->normals[nCounter + i + 2] = vN.z;

                mesh->normals[nCounter + i + 3] = vN.x;
                mesh->normals[nCounter + i + 4] = vN.y;
                mesh->normals[nCounter + i + 5] = vN.z;

                mesh->normals[nCounter + i + 6] = vN.x;
                mesh->normals[nCounter + i + 7] = vN.y;
                mesh->normals[nCounter + i + 8] = vN.z;
            }

            nCounter += 18;     // 6 vertex, 18 floats
        }
    }

    // Upload vertex data to GPU (static mesh)
    UpdateMeshBuffer(*mesh, 0, mesh->vertices, mesh->vertexCount * 3 * sizeof(float), 0); // Update vertices
    UpdateMeshBuffer(*mesh, 2, mesh->normals, mesh->vertexCount * 3 * sizeof(float), 0); // Update normals
}


// Required function. Should handle creating the client on first call
void c_render(Battle* env) {
    if (env->client == NULL) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(env->width, env->height, "PufferLib Battle");
        SetTargetFPS(30);
        Client* client = (Client*)calloc(1, sizeof(Client));
        env->client = client;
        client->models[DRONE] = LoadModel("resources/battle/drone.glb");
        client->models[FIGHTER] = LoadModel("resources/battle/fighter.glb");
        client->models[MOTHERSHIP] = LoadModel("resources/battle/mothership.glb");
        client->models[BOMBER] = LoadModel("resources/battle/bomber.glb");
        client->models[INFANTRY] = LoadModel("resources/battle/car.glb");
        client->models[TANK] = LoadModel("resources/battle/tank.glb");
        client->models[ARTILLERY] = LoadModel("resources/battle/artillery.glb");
        client->models[BASE] = LoadModel("resources/battle/base.glb");
        //env->client->ship = LoadModel("resources/puffer.glb");
        
        char vsPath[256];
        char fsPath[256];
        sprintf(vsPath, "resources/tower_climb/shaders/gls%i/lighting.vs", GLSL_VERSION);
        sprintf(fsPath, "resources/tower_climb/shaders/gls%i/lighting.fs", GLSL_VERSION);
        client->light_shader = LoadShader(vsPath, fsPath);
        client->light = CreateLight(LIGHT_DIRECTIONAL, 
            (Vector3){ 0.0f, 10.0f, 0.0f },    // High above for top lighting
            (Vector3){ 0.5f, -1.0f, 0.3f },    // Direction: down and slightly forward
            (Color){ 180, 180, 190, 255 },    // Softer warm white for tops
            client->light_shader);

        for (int i = 0; i < 8; i++) {
            Model* m = &client->models[i];
            for (int j = 0; j < m->materialCount; j++) {
                //m->materials[j].maps[MATERIAL_MAP_DIFFUSE].texture = client->vehicle_texture;
                m->materials[j].shader = client->light_shader;
            }
        }
 
        Camera3D camera = { 0 };
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
        camera.fovy = 45.0f;                                // Camera field-of-view Y
        camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
        camera.position = (Vector3){ 0, 5*env->size_y, -3*env->size_z};
        camera.target = (Vector3){ 0, 0, 0};
        client->camera = camera;

        client->mesh = create_heightmap_mesh(env->terrain, (Vector3){env->terrain_width, 1, env->terrain_height});
        client->model = LoadModelFromMesh(*client->mesh);
        update_heightmap_mesh(client->mesh, env->terrain, (Vector3){env->terrain_width, 1, env->terrain_height});

        client->terrain_shader = LoadShader(
            TextFormat("resources/battle/shader_%i.vs", GLSL_VERSION),
            TextFormat("resources/battle/shader_%i.fs", GLSL_VERSION)
        );

        Image img = GenImageColor(env->terrain_width, env->terrain_height, WHITE);
        ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        client->terrain_texture = LoadTextureFromImage(img);
        UnloadImage(img);

        client->terrain_shader_loc = GetShaderLocation(client->terrain_shader, "terrain");
        SetShaderValueTexture(client->terrain_shader, client->terrain_shader_loc, client->terrain_texture);

        client->terrain_data = calloc(4*env->terrain_width*env->terrain_height, sizeof(unsigned char));
        for (int i = 0; i < env->terrain_width*env->terrain_height; i++) {
            client->terrain_data[4*i] = env->terrain[i];
            client->terrain_data[4*i+3] = 255;
        }
        UpdateTexture(client->terrain_texture, client->terrain_data);
        SetShaderValueTexture(client->terrain_shader, client->terrain_shader_loc, client->terrain_texture);

        int shader_width_loc = GetShaderLocation(client->terrain_shader, "width");
        SetShaderValue(client->terrain_shader, shader_width_loc, &env->terrain_width, SHADER_UNIFORM_INT);

        int shader_height_loc = GetShaderLocation(client->terrain_shader, "height");
        SetShaderValue(client->terrain_shader, shader_height_loc, &env->terrain_height, SHADER_UNIFORM_INT);
 
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    Client* client = env->client;
    UpdateCamera(&client->camera, CAMERA_THIRD_PERSON);
    //UpdateLightValues(client->light);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode3D(client->camera);

        //BeginShaderMode(client->terrain_shader);
        client->model.materials[0].shader = client->terrain_shader;
        Vector3 pos = {-env->size_x, -env->size_y, -env->size_z};
        DrawModel(client->model, pos, 1.0/128.0f, (Color){156, 50, 20, 255});
        //EndShaderMode();


        for (int f=0; f<env->num_armies; f++) {
            Entity* base = &env->bases[f];
            float y = ground_height(env, base->x, base->z);
            DrawModel(client->models[BASE], (Vector3){base->x, y, base->z}, 0.05f, COLORS[base->army]);
        }

        for (int i=0; i<env->num_agents; i++) {
            Entity* agent = &env->agents[i];

            Vector3 pos = {agent->x, agent->y, agent->z};
            Matrix transform = QuaternionToMatrix(agent->orientation);
            Model model = client->models[agent->unit];
            model.transform = transform;

            Vector3 scale = (Vector3){0.01f, 0.01f, 0.01f};
            if (agent->unit == DRONE) {
                scale = (Vector3){0.01f, 0.01f, 0.01f};
            } else if (agent->unit == MOTHERSHIP) {
                scale = (Vector3){0.03f, 0.03f, 0.03f};
            } else if (agent->unit == FIGHTER) {
                scale = (Vector3){0.015f, 0.015f, 0.015f};
            } else if (agent->unit == BOMBER) {
                scale = (Vector3){0.015f, 0.015f, 0.015f};
            } else if (agent->unit == INFANTRY) {
                scale = (Vector3){0.005f, 0.005f, 0.005f};
            } else if (agent->unit == TANK) {
                scale = (Vector3){0.01f, 0.01f, 0.01f};
            } else if (agent->unit == ARTILLERY) {
                scale = (Vector3){0.02f, 0.02f, 0.02f};
            }

            Color color = COLORS[agent->army];
            Vector3 rot = {0.0f, 1.0f, 0.0f};
            DrawModelEx(model, pos, rot, 0, scale, color);

            if (agent->target >= 0) {
                Entity* target = &env->agents[agent->target];
                DrawLine3D(
                    (Vector3){agent->x, agent->y, agent->z},
                    (Vector3){target->x, target->y, target->z},
                    COLORS[agent->army]
                );
            }
        }

        DrawCubeWires(
            (Vector3){0, 0, 0},
            2*env->size_x, 2*env->size_y, 2*env->size_z,
            (Color){0, 255, 255, 128}
        );

    EndMode3D();
    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Battle* env) {
    free(env->agents);
    free(env->bases);
    if (env->client != NULL) {
        Client* client = env->client;
        //UnloadTexture(client->sprites);
        CloseWindow();
        free(client);
    }
}
