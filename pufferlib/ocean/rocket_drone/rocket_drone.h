// Standalone C demo for RocketDrone environment
// Compile using: ./scripts/build_ocean.sh rocket_drone [local|fast]
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

// Task system removed - pure combat mode only
#define MAX_ROCKETS 100
#define ROCKET_DMG 100
#define ROCKET_VEL 15.0f

#define HIT_REWARD 5.f
#define HIT_PUNISH 7.f
#define DEATH_PUNISH 10.f


// Task names removed - no task system

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

    int num_agents;
    Drone* agents;

    Rocket rockets[MAX_ROCKETS];
    int rocket_count;

    Client *client;
} RocketDrone;
static bool player_active = false;
static int player_idx = -1;
static float player_yaw = 0.0f, player_pitch = 0.0f;
static void player_character(RocketDrone* env, int idx) {
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

void init(RocketDrone *env) {
    env->agents = calloc(env->num_agents, sizeof(Drone));
    env->log = (Log){0};
    env->tick = 0;
}

void add_log(RocketDrone *env, int idx, bool oob) {
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

Drone* nearest_drone(RocketDrone* env, Drone *agent) {
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

void compute_observations(RocketDrone *env) {
    int idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        Quat q_inv = quat_inverse(agent->quat);
        Vec3 linear_vel_body = quat_rotate(q_inv, agent->vel);
        Vec3 drone_up_world = quat_rotate(agent->quat, (Vec3){0.0f, 0.0f, 1.0f});

        // Safe_obs = somewhat hacky way of avoiding numerical errors in training
        
        env->observations[idx++] = safe_obs(safe_div(linear_vel_body.x, agent->max_vel, 0.0f));
        env->observations[idx++] = safe_obs(safe_div(linear_vel_body.y, agent->max_vel, 0.0f));
        env->observations[idx++] = safe_obs(safe_div(linear_vel_body.z, agent->max_vel, 0.0f));

        env->observations[idx++] = safe_obs(safe_div(agent->omega.x, agent->max_omega, 0.0f));
        env->observations[idx++] = safe_obs(safe_div(agent->omega.y, agent->max_omega, 0.0f));
        env->observations[idx++] = safe_obs(safe_div(agent->omega.z, agent->max_omega, 0.0f));

        env->observations[idx++] = safe_obs(drone_up_world.x);
        env->observations[idx++] = safe_obs(drone_up_world.y);
        env->observations[idx++] = safe_obs(drone_up_world.z);

        env->observations[idx++] = safe_obs(agent->quat.w);
        env->observations[idx++] = safe_obs(agent->quat.x);
        env->observations[idx++] = safe_obs(agent->quat.y);
        env->observations[idx++] = safe_obs(agent->quat.z);

        env->observations[idx++] = safe_obs(safe_div(agent->rpms[0], agent->max_rpm, 0.0f));
        env->observations[idx++] = safe_obs(safe_div(agent->rpms[1], agent->max_rpm, 0.0f));
        env->observations[idx++] = safe_obs(safe_div(agent->rpms[2], agent->max_rpm, 0.0f));
        env->observations[idx++] = safe_obs(safe_div(agent->rpms[3], agent->max_rpm, 0.0f));

        env->observations[idx++] = safe_obs(agent->pos.x / GRID_X);
        env->observations[idx++] = safe_obs(agent->pos.y / GRID_Y);
        env->observations[idx++] = safe_obs(agent->pos.z / GRID_Z);

        env->observations[idx++] = safe_obs(agent->spawn_pos.x / GRID_X);
        env->observations[idx++] = safe_obs(agent->spawn_pos.y / GRID_Y);
        env->observations[idx++] = safe_obs(agent->spawn_pos.z / GRID_Z);

        // No target following - zero out target observations
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;

        env->observations[idx++] = safe_obs(agent->last_collision_reward);
        env->observations[idx++] = safe_obs(agent->last_target_reward);
        env->observations[idx++] = safe_obs(agent->last_abs_reward);

        // Multiagent obs
        Drone* nearest = nearest_drone(env, agent);
        if (env->num_agents > 1 && nearest != NULL) {
            env->observations[idx++] = safe_obs(clampf(nearest->pos.x - agent->pos.x, -1.0f, 1.0f));
            env->observations[idx++] = safe_obs(clampf(nearest->pos.y - agent->pos.y, -1.0f, 1.0f));
            env->observations[idx++] = safe_obs(clampf(nearest->pos.z - agent->pos.z, -1.0f, 1.0f));
        } else {
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
            env->observations[idx++] = 0.0f;
        }

        // rocket obs: firing status, rocket count, nearest (enemy) rocket.
        float cooldown_norm = safe_div((float)agent->rocket_cooldown, (float)ROCKET_COOLDOWN, 0.0f);
        float rockets_norm = safe_div((float)env->rocket_count, (float)MAX_ROCKETS, 0.0f);
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
            if (d < nearest_dist && d > 1e-8f) {
                nearest_dist = d;
                Vec3 dir_world = {dx, dy, dz};
                float inv_d = safe_div(1.0f, d, 0.0f);
                dir_world.x *= inv_d;
                dir_world.y *= inv_d;
                dir_world.z *= inv_d;
                dir_body = quat_rotate(q_inv, dir_world);
            }
        }
        if (nearest_dist < MAX_DIST) dist_norm = safe_div(nearest_dist, MAX_DIST, 0.0f);
        env->observations[idx++] = safe_obs(cooldown_norm);
        env->observations[idx++] = safe_obs(rockets_norm);
        env->observations[idx++] = safe_obs(dir_body.x);
        env->observations[idx++] = safe_obs(dir_body.y);
        env->observations[idx++] = safe_obs(dir_body.z);
        env->observations[idx++] = safe_obs(dist_norm);
    }
    
}

// All target-setting functions removed - pure combat mode

float compute_reward(RocketDrone* env, Drone *agent, bool collision) {
    // Only collision penalties for physical crashes, no formation rewards
    float collision_penalty = 0.0f;
    if (collision && env->num_agents > 1) {
        Drone *nearest = nearest_drone(env, agent);
        if (nearest != NULL) {
            float dx = agent->pos.x - nearest->pos.x;
            float dy = agent->pos.y - nearest->pos.y;
            float dz = agent->pos.z - nearest->pos.z;
            float min_dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (min_dist < 1.0f) {
                collision_penalty = -0.1f;
                agent->collisions += 1.0f;
            }
        }
    }

    agent->last_collision_reward = collision_penalty;
    agent->last_target_reward = 0.0f;
    agent->last_abs_reward = collision_penalty;

    agent->episode_length++;
    agent->score += collision_penalty;

    return collision_penalty;
}

void reset_agent(RocketDrone* env, Drone *agent, int idx) {
    agent->uid = uid_tracker++;
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->health = 100;


    // spawn new drones in a sphere centered at origin, touching close to env boundaries
    float u1 = rndf(0, 1);
    float u2 = rndf(0, 1);
    float z_norm = 1 - 2 * u1;
    float phi = 2 * M_PI * u2;
    float r_norm = sqrtf(1 - z_norm * z_norm);
    float x_norm = r_norm * cosf(phi);
    float y_norm = r_norm * sinf(phi);
    
    float sphere_radius = rndf(7, 9);
    Vec3 sphere_center = {0, 0, GRID_Z / 2.0f};
    agent->pos = (Vec3){
        sphere_center.x + x_norm * sphere_radius,
        sphere_center.y + y_norm * sphere_radius, 
        sphere_center.z + z_norm * sphere_radius
    };
    agent->spawn_pos = agent->pos;
    agent->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->omega = (Vec3){0.0f, 0.0f, 0.0f};
    agent->quat = (Quat){1.0f, 0.0f, 0.0f, 0.0f};
    agent->target_pos = (Vec3){0.0f, 0.0f, 0.0f};
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};

    float size = rndf(0.1f, 0.4);
    init_drone(agent, size, 0.1f);
    compute_reward(env, agent, true);
}

void c_reset(RocketDrone *env) {
    // printf("RESET: Full environment reset (episode timeout or manual)\n");
    env->tick = 0;
    
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        // printf("RESET: Agent %d - Full environment reset\n", i);
        reset_agent(env, agent, i);
    }
 
    compute_observations(env);
}

// maps drone unique identifier to env array identifier
int drone_index_lookup(RocketDrone *env, int uid) {
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


void update_rockets(RocketDrone *env) {

    for (int r = 0; r < env->rocket_count; r++) {  // increment vel + oob / collision check
        Rocket *rocket = &env->rockets[r];
        rocket->pos = add3(rocket->pos, scalmul3(rocket->vel, DT));
        
        if (rocket->pos.x < -GRID_X || rocket->pos.x > GRID_X ||
            rocket->pos.y < -GRID_Y || rocket->pos.y > GRID_Y ||
            rocket->pos.z < 0.0f || rocket->pos.z > GRID_Z) {
            env->rockets[r] = env->rockets[--env->rocket_count]; r--; continue;
        }
    
        for (int d = 0; d < env->num_agents; d++) {    
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
                    printf("RESET: Agent %d - Destroyed by rocket from Agent %d\n", 
                            target_idx, shooter_idx);
                    env->rewards[target_idx] -= DEATH_PUNISH;
                    env->terminals[target_idx] = 1;

                }
                env->rockets[r] = env->rockets[env->rocket_count - 1];
                env->rocket_count--;
            }

        }
    }
}









void c_step(RocketDrone *env) {
    env->tick++;
    update_rockets(env);
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;

        float* atn = &env->actions[7*i];   // 4 movement actions + 3 rocket (fire bool + angles)
        
        
        move_drone(agent, atn);

        
        if(atn[4]>0.f && env->rocket_count<MAX_ROCKETS && agent->rocket_cooldown <= 0){
            // horizontal angle
            float az_norm = atn[5] - 2.0f * floorf((atn[5] + 1.0f) / 2.0f);
            float az = az_norm * M_PI;
            
            // vertical angle
            float el = clampf(atn[6], -1.0f, 1.0f) * 1.57079632679f;
            Vec3 dir={cosf(el)*cosf(az),cosf(el)*sinf(az),sinf(el)};
            Rocket r={agent->pos,scalmul3(dir,ROCKET_VEL),agent};
            env->rockets[env->rocket_count++]=r;
            agent->rocket_cooldown = ROCKET_COOLDOWN;
        }

        
        bool out_of_bounds = agent->pos.x < -GRID_X || agent->pos.x > GRID_X ||
                             agent->pos.y < -GRID_Y || agent->pos.y > GRID_Y ||
                             agent->pos.z <  0.0f   || agent->pos.z > GRID_Z;

        
        float reward = compute_reward(env, agent, true);

        env->rewards[i] += reward;
        agent->episode_return += reward;

        if (out_of_bounds) {
            // printf("RESET: Agent %d - Out of bounds (pos: %.1f, %.1f, %.1f)\n", 
            //        i, agent->pos.x, agent->pos.y, agent->pos.z);
            env->rewards[i] -= 5;
            env->terminals[i] = 1;
            add_log(env, i, true);
            reset_agent(env, agent, i);
        } else if (env->tick >= HORIZON - 1) {
            // printf("RESET: Agent %d - Episode timeout (will reset next step)\n", i);
            env->terminals[i] = 1;
            add_log(env, i, false);
        }

        if (env->agents[i].rocket_cooldown > 0) {
            env->agents[i].rocket_cooldown--;
        }
    }
    


    if (env->tick >= HORIZON - 1) {
        // printf("RESET: Episode timeout reached (tick %d >= %d)\n", env->tick, HORIZON - 1);
        c_reset(env);
    }

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(RocketDrone *env) {
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

    c->camera.position = (Vector3){x, y, z + GRID_Z / 2.0f};
    c->camera.target = (Vector3){0, 0, GRID_Z / 2.0f};
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

Client *make_client(RocketDrone *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    InitWindow(WIDTH, HEIGHT, "PufferLib RocketDrone");

#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->camera_distance = 30.0f;
    client->camera_azimuth = 0.0f;
    client->camera_elevation = PI / 6.0f;
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


void draw_rockets(RocketDrone *env) {
    for(int i = 0; i < env->rocket_count; i++) {
        Rocket *rocket = &env->rockets[i];
        DrawSphere((Vector3){rocket->pos.x, rocket->pos.y, rocket->pos.z}, 0.1f, RED);
    }
}

void c_render(RocketDrone *env) {
    static int last_tick = -1;
    

    last_tick = env->tick;
    
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

    Client *client = env->client;


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

 
    DrawCubeWires((Vector3){0.0f, 0.0f, GRID_Z / 2.0f}, GRID_X * 2.0f,
        GRID_Y * 2.0f, GRID_Z, WHITE);
    draw_rockets(env);
    for (int i = 0; i < env->num_agents; i++) {
        if (player_active && i == player_idx) continue;
        Drone *agent = &env->agents[i];

   
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
    DrawText("Mode: Pure Combat", 10, 50, 16, PUFF_WHITE);

    EndDrawing();
}


