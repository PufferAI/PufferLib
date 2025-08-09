#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include <time.h>

#define FIRE 0
#define LEFT 1
#define RIGHT 2
#define UP 3
#define DOWN 4
#define FUSETUP 5
#define FUSETDOWN 6

#define MAX_PROJECTILE_TIME 60.0f
#define TIMESTEP 0.25f

typedef struct Log {
    float score;
    float episode_return;
    float episode_length;
    float dist;
    float max_reward_distn;
    float turn_penaltyn;
    float acc1000;
    float n;
} Log;

typedef struct Client {
    Camera3D camera;
    float x_size;
    float y_size;
    float z_size;
    int render;
    int debug;
    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
} Client;

typedef struct Artillery3D {
    Client* client;
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    int i;

    int x_size;
    int y_size;
    int z_size;
    float score;
    int tick;
    float dist;

    int moving_target;

    float muzzle_v;
    float fuse_t;
    float azimuth;
    float azimuth0;
    float elevation;
    float elevation0;

    float px;
    float py;
    float pz;
    float g;
    int projectile_active;
    float projectile_time;

    float v0;
    float vx0;
    float vy0;
    float vz0;
    float x0;
    float y0;
    float z0;

    float target_min_x;
    float target_max_x;
    float target_min_y;
    float target_max_y;
    float target_min_z;
    float target_max_z;
    float target_size;
    float target_vx;
    float target_vy;
    float target_vz;
    float tx;
    float ty;
    float tz;

    float max_reward;
    float max_reward_dist;
    float max_reward_distn;
    float max_dist0;
    float dist_fade;
    float turn_penalty;
    float turn_penaltyn;
    int turn_penalty_delay;
    float turn_penalty_ramp;
    float miss_penalty;
    int fired;
    float out_bounds_penalty;

    int frameskip;
    int render;
    int continuous;

    int debug;
    unsigned int rng;
    int same_runs;
    int runs;

    // Math
    float inv_x_size;
    float inv_y_size;
    float inv_z_size;
} Artillery3D;

void c_close(Artillery3D* env) {
}

void free_allocated(Artillery3D* env) {
    free(env->actions);
    free(env->observations);
    free(env->terminals);
    free(env->rewards);
    c_close(env);
}

void add_log(Artillery3D* env) {
    env->log.episode_length += env->tick;
    env->log.episode_return += env->score;
    env->log.score += env->score;
    env->log.dist += env->dist;
    env->log.max_reward_distn += env->max_reward_distn;
    env->log.turn_penaltyn += env->turn_penaltyn;
    env->log.n += 1;
    env->log.acc1000 += 750.0f - env->dist;
}

void calculate_parabola_closest_distance(Artillery3D* env) {
    env->v0 = env->muzzle_v;
    env->vx0 = env->v0 * cosf(env->azimuth);
    env->vy0 = env->v0 * sinf(env->azimuth);
    env->vz0 = env->v0 * sinf(env->elevation);
    env->x0 = 30.0f;
    env->y0 = 30.0f;
    env->z0 = 30.0f;

    float tx = env->tx;
    float ty = env->ty;
    float tz = env->tz;
    float tvx = env->target_vx;
    float tvy = env->target_vy;
    float tvz = env->target_vz;
    float txn = tx;
    float tyn = ty;
    float tzn = tz;

    float t = env->fuse_t;

    float x = env->x0 + env->vx0 * t;
    float y = env->y0 + env->vy0 * t;
    float z = env->z0 + env->vz0 * t - 0.5f * env->g * t * t;

    txn = tx + tvx * t;
    tyn = ty + tvy * t;
    tzn = tz + tvz * t;

    float dx = x - txn;
    float dy = y - tyn;
    float dz = z - tzn;

    env->dist = sqrt(dx * dx + dy * dy + dz * dz);
}

void fire_projectile(Artillery3D* env) {
    if (env->debug > 0) printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!FIRE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    calculate_parabola_closest_distance(env);
    float score;

    if (env->dist >= env->max_reward_distn) {
        score = env->miss_penalty;
    } else {
        score = 1.0f - (env->dist / env->max_reward_distn);
    }

    if (env->debug > 0) printf("    env%d tick%d closest_dist = %.3f score=%.3f\n", env->i, env->tick, env->dist, score);
    env->score += score;
    env->rewards[0] += score;

    if (env->render) {
        env->projectile_active = 1;
        env->projectile_time = 0.0f;
        env->px = env->x0;
        env->py = env->y0;
        env->pz = env->z0;
    }
}

void compute_observations(Artillery3D* env) {
    if (env->debug > 0) printf("  Compute Observations\n");

    env->observations[0] = env->azimuth;
    if (env->debug > 0) printf("    azimuth = %.3f\n", env->observations[0]);

    env->observations[1] = env->elevation;
    if (env->debug > 0) printf("    elevation = %.3f\n", env->observations[1]);

    env->observations[2] = env->tx * env->inv_x_size;
    if (env->debug > 0) printf("    tx = %.3f\n", env->observations[2]);

    env->observations[3] = env->ty * env->inv_y_size;
    if (env->debug > 0) printf("    ty = %.3f\n", env->observations[3]);

    env->observations[4] = env->tz * env->inv_z_size;
    if (env->debug > 0) printf("    tz = %.3f\n", env->observations[4]);

    env->observations[5] = env->score;
    if (env->debug > 0) printf("    score = %.6f\n", env->observations[5]);

    env->observations[6] = env->tick * 0.01;
    if (env->debug > 0) printf("    tick = %.3f\n", env->observations[6]);

    env->observations[7] = env->target_vx * 0.01;
    if (env->debug > 0) printf("    target_vx = %.3f\n", env->observations[7]);

    env->observations[8] = env->target_vy * 0.01;
    if (env->debug > 0) printf("    target_vy = %.3f\n", env->observations[8]);

    env->observations[9] = env->target_vz * 0.01;
    if (env->debug > 0) printf("    target_vz = %.3f\n", env->observations[9]);

    env->observations[10] = env->fuse_t;
    if (env->debug > 0) printf("    fuse_t = %.3f\n", env->observations[10]);

    //printf("H Obs: ");
    //for(int i = 0; i < 1; i++) {
    //    printf("%.3f ", env->observations[i]);
    //}
    //printf("\n");
}

static inline float clampf(float v, float min, float max) {
    if (v < min)
        return min;
    if (v > max)
        return max;
    return v;
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

        float sensitivity = 0.005f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

Client* make_client(Artillery3D* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->x_size = env->x_size;
    client->y_size = env->y_size;
    client->z_size = env->z_size;

    InitWindow(env->x_size, env->y_size, "PufferLib Artillery3D");
    SetTargetFPS(30);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
}

void get_random_start(Artillery3D* env) {
    if (env->debug > 0) printf("get_random_start\n");
    env->tx = rand() % env->x_size;
    env->ty = rand() % env->y_size;
    env->tz = rand() % env->z_size;
    if (env->tx < env->target_min_x) env->tx = env->target_min_x;
    if (env->tx > env->target_max_x) env->tx = env->target_max_x;
    if (env->ty < env->target_min_y) env->ty = env->target_min_y;
    if (env->ty > env->target_max_y) env->ty = env->target_max_y;
    if (env->tz < env->target_min_z) env->tz = env->target_min_z;
    if (env->tz > env->target_max_z) env->tz = env->target_max_z;
    env->azimuth = (float)rand() / (float)RAND_MAX;
    env->azimuth0 = env->azimuth;
    env->elevation = (float)rand() / (float)RAND_MAX;
    env->elevation0 = env->elevation;
    env->target_vx = -rand() % 20 - 30;
    env->target_vy = -rand() % 20 - 30;
    env->target_vz = -rand() % 5 - 5;
}

void reset_round(Artillery3D* env) {
    env->terminals[0] = 0;
    if (env->runs % (int)env->same_runs == 0) {
        get_random_start(env);
    }
    else {
        env->azimuth = env->azimuth0;
        env->elevation = env->elevation0;
    }
    env->tick = 0;
    env->runs += 1;
    env->score = 0;
    env->fired = 0;
    env->projectile_active = 0;
    env->projectile_time = 0.0f;
    env->max_reward_distn = env->max_dist0 - (int)(env->runs * env->dist_fade);
    if (env->max_reward_distn < env->max_reward_dist) env->max_reward_distn = env->max_reward_dist;
}

void c_reset(Artillery3D* env) {
    compute_observations(env);
    reset_round(env);
}

void c_render(Artillery3D* env) {
    env->render = 1;
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }

    if (WindowShouldClose()) {
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    
    if (IsKeyPressed(KEY_TAB)) {
        ToggleFullscreen();
    }

    handle_camera_controls(env->client);

    BeginDrawing();
    ClearBackground((Color){135, 206, 235, 255});

    BeginMode3D(env->client->camera);

    DrawPlane((Vector3){0.0f, 0.0f, 0.0f}, (Vector2){env->x_size * 2.0f, env->y_size * 2.0f}, (Color){34, 139, 34, 255});
    
    DrawCubeWires((Vector3){env->x_size / 2.0f, env->y_size / 2.0f, env->z_size / 2.0f}, 
                  env->x_size, env->y_size, env->z_size, WHITE);

    DrawSphere((Vector3){env->tx, env->ty, env->tz}, env->target_size, RED);

    float barrel_length = 40.0f;
    float barrel_radius = 4.0f;
    Vector3 cannon_pos = {30.0f, 30.0f, 30.0f};

    float azimuth = env->azimuth;
    float elevation = env->elevation;
    
    Vector3 barrel_direction = {
        cosf(azimuth) * cosf(elevation),
        sinf(azimuth) * cosf(elevation),
        sinf(elevation)
    };

    Vector3 barrel_end = {
        cannon_pos.x + barrel_length * barrel_direction.x,
        cannon_pos.y + barrel_length * barrel_direction.y,
        cannon_pos.z + barrel_length * barrel_direction.z
    };

    DrawCylinderEx(cannon_pos, barrel_end, barrel_radius, barrel_radius, 12, DARKGRAY);
    
    DrawSphere(cannon_pos, 12.0f, GRAY);

    float v0 = env->muzzle_v;
    float vx0 = v0 * barrel_direction.x;
    float vy0 = v0 * barrel_direction.y;
    float vz0 = v0 * barrel_direction.z;
    
    Vector3 prev_point = cannon_pos;
    int segment_count = 0;
    
    for (float t = TIMESTEP; t < MAX_PROJECTILE_TIME; t += TIMESTEP) {
        float x = cannon_pos.x + vx0 * t;
        float y = cannon_pos.y + vy0 * t;
        float z = cannon_pos.z + vz0 * t - 0.5f * env->g * t * t;

        if (z < 0 || x < 0 || x > env->x_size || y < 0 || y > env->y_size) break;

        Vector3 current_point = {x, y, z};
        
        Color trajectory_color = (segment_count % 2 == 0) ? WHITE : LIGHTGRAY;
        DrawLine3D(prev_point, current_point, trajectory_color);
        
        prev_point = current_point;
        segment_count++;
    }

    if (env->projectile_active) {
        DrawSphere((Vector3){env->px, env->py, env->pz}, 4.0f, BLACK);
    }

    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, WHITE);
    DrawText("Tab: Toggle fullscreen", 10, 50, 16, WHITE);
    DrawText(TextFormat("Score: %.3f", env->score), 10, 70, 20, BLACK);

    EndDrawing();
}

void init(Artillery3D* env) {
    env->runs = 0;
    env->tick = 0;
    if (env->same_runs < 1) env->same_runs = 1;
    env->g = 9.8f;
    env->projectile_active = 0;
    env->projectile_time = 0.0f;
    env->dist = env->x_size;

    env->inv_x_size = 1.0f / env->x_size;
    env->inv_y_size = 1.0f / env->y_size;
    env->inv_z_size = 1.0f / env->z_size;

    srand(env->rng + env->i);

    get_random_start(env);
}

void allocate(Artillery3D* env) {
    init(env);
    env->observations = (float*)calloc(11, sizeof(float));
    env->actions = (float*)calloc(1, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}

float get_turn_penalty(Artillery3D* env) {
    int start_tick = env->turn_penalty_delay;
    int end_tick = start_tick + env->turn_penalty_ramp;

    if (env->tick <= start_tick) {
        return 0.0f;
    } else if (env->tick < end_tick) {
        float progress = (env->tick - start_tick) * env->turn_penalty_ramp;
        return env->turn_penalty * progress;
    } else {
        return env->turn_penalty;
    }
}

void step_frame(Artillery3D* env, float action) {
    if (env->debug > 0) printf("STEP env%d tick%d=========================\n", env->i, env->tick);

    if (!env->projectile_active) {
        if (action == FIRE) {
            env->fired = 1;
            if (env->projectile_active == 0) {
                fire_projectile(env);
            }
        } else if (action == LEFT) {
            if (env->azimuth < 0.95) {
                env->azimuth += 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == RIGHT) {
            if (env->azimuth > 0.05) {
                env->azimuth -= 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == UP) {
            if (env->elevation < 0.95) {
                env->elevation += 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == DOWN) {
            if (env->elevation > 0.05) {
                env->elevation -= 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == FUSETUP) {
            if (env->fuse_t < 0.95) {
                env->fuse_t += 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == FUSETDOWN) {
            if (env->fuse_t > 0.05) {
                env->fuse_t -= 0.05;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        }

        if (action != FIRE) {
            env->turn_penaltyn = get_turn_penalty(env);
            env->score += env->turn_penaltyn;
            env->rewards[0] += env->turn_penaltyn;
        }
    }
    else { // Projectile Active
        env->projectile_time += TIMESTEP;
        env->px = env->x0 + env->vx0 * env->projectile_time;
        if (env->debug > 1) printf("env->px = %.3f, env->vx0 = %.3f, ptime = %.3f\n", env->px, env->vx0, env->projectile_time);
        if (env->debug > 1) printf("env->tx = %.3f\n", env->tx);
        env->py = env->y0 + env->vy0 * env->projectile_time;
        env->pz = env->z0 + env->vz0 * env->projectile_time - 0.5f * env->g * env->projectile_time * env->projectile_time;
        if (env->debug > 1) printf("env->py = %.3f, env->vy0 = %.3f, ptime = %.3f\n", env->py, env->vy0, env->projectile_time);
    }

    if (env->moving_target == 1) {
        env->tx += env->target_vx * TIMESTEP;
        env->ty += env->target_vy * TIMESTEP;
        env->tz += env->target_vz * TIMESTEP;
    }

    if (env->debug > 1) printf("  env->px = %.1f env->tx = %.1f env->render=%d\n", env->px, env->tx, env->render);
    if ((env->fired == 1 && (!env->render || env->px > env->tx + env->target_size || env->py < 0.0f)) || (env->score < -1.0f)) {
        if (env->debug > 0) printf("==================terminate=================\n\n\n\n\n\n\n\n\n\n");
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
}

void c_step(Artillery3D* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    float action = env->actions[0];
    for (int i = 0; i < env->frameskip; i++) {
        env->tick += 1;
        step_frame(env, action);
    }
    compute_observations(env);
}
