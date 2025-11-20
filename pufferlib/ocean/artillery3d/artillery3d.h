#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "raylib.h"
#include "rlgl.h"
#include <time.h>

#define FIRE 0
#define LEFT 1
#define RIGHT 2
#define UP 3
#define DOWN 4
#define FUSETUP 5
#define FUSETDOWN 6

#define TIMESTEP 0.0333f
#define MUZZLEV 850.0f
#define FUSEMULT 2.0f // Fuse Time Multiplier
#define EXPLRAD 20.0f

#define XSIZE 2000
#define YSIZE 1000
#define ZSIZE 200

#define X0 0.0f
#define Y0 YSIZE * 0.5f
#define Z0 0.0f
#define G 9.81f

#define TXMIN 1500 // Target X Min
#define TXMAX 1900
#define TYMIN 250
#define TYMAX 750
#define TZMIN 10
#define TZMAX 100

#define NUMTARGETS 2
#define MAXSHOTS 4

typedef struct Log {
    float perf;
    float score;
    float episode_length;
    float dist;
    float max_reward_distn;
    float turn_penaltyn;
    float acc1000;
    float n;
    float shots_fired;
    float targets_remaining;
} Log;

typedef struct Client {
    Camera3D camera;
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
    int* actions;
    float* rewards;
    unsigned char* terminals;

    float score;
    int tick;
    float dist;

    float fuse_t;
    float fuse_t0;
    float azimuth;
    float azimuth0;
    float elevation;
    float elevation0;
    float t;
    float fire_t;

    float px;
    float py;
    float pz;
    int projectile_active;
    float projectile_time;
    float boom_x;
    float boom_y;
    float boom_z;
    int shots_fired;
    int shots_remaining;

    float v0;
    float vx0;
    float vy0;
    float vz0;

    float target_size;
    float* target_vx;
    float* target_vy;
    float* target_vz;
    float* tx;
    float* ty;
    float* tz;
    int targets_remaining;
    float* time_target_vanish;
    float hit;

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

    int render;

    int runs;

    float inv_x_size;
    float inv_y_size;
    float inv_z_size;
    float inv_max_shots;
} Artillery3D;

void get_random_start(Artillery3D* env) {
    for (int i = 0; i < NUMTARGETS; i++) {
        env->tx[i] = (rand() % (TXMAX - TXMIN)) + TXMIN;
        env->ty[i] = (rand() % (TYMAX - TYMIN)) + TYMIN;
        env->tz[i] = (rand() % (TZMAX - TZMIN)) + TZMIN;
        if (env->tx[i] < TXMIN) env->tx[i] = TXMIN;
        if (env->tx[i] > TXMAX) env->tx[i] = TXMAX;
        if (env->ty[i] < TYMIN) env->ty[i] = TYMIN;
        if (env->ty[i] > TYMAX) env->ty[i] = TYMAX;
        if (env->tz[i] < TZMIN) env->tz[i] = TZMIN;
        if (env->tz[i] > TZMAX) env->tz[i] = TZMAX;

        env->target_vx[i] = -rand() % 50 - 30;
        env->target_vy[i] = -rand() % 30 - 20;
        env->target_vz[i] = -rand() % 5 - 5;
        if (env->ty[i] < Y0) env->target_vy[i] = env->target_vy[i] * -1.0f;
    }
    env->azimuth = 0.5f;
    env->azimuth0 = env->azimuth;
    env->elevation = 0.0f;
    env->elevation0 = env->elevation;
    env->fuse_t = 0.7f;
    env->fuse_t0 = env->fuse_t;
}

void init(Artillery3D* env) {
    env->runs = 0;
    env->tick = 0;
    env->t = 0.0f;
    env->fire_t = 99999.0f;
    env->projectile_active = 0;
    env->projectile_time = 0.0f;
    env->dist = XSIZE;

    env->inv_x_size = 1.0f / XSIZE;
    env->inv_y_size = 1.0f / YSIZE;
    env->inv_z_size = 1.0f / ZSIZE;
    env->inv_max_shots = 1.0f / MAXSHOTS;

    env->tx = (float*)calloc(NUMTARGETS, sizeof(float));
    env->ty = (float*)calloc(NUMTARGETS, sizeof(float));
    env->tz = (float*)calloc(NUMTARGETS, sizeof(float));
    env->target_vx = (float*)calloc(NUMTARGETS, sizeof(float));
    env->target_vy = (float*)calloc(NUMTARGETS, sizeof(float));
    env->target_vz = (float*)calloc(NUMTARGETS, sizeof(float));
    env->time_target_vanish = (float*)calloc(NUMTARGETS, sizeof(float));

    get_random_start(env);
}

static inline float get_turn_penalty(Artillery3D* env) {
    int start_tick = env->turn_penalty_delay;

    if (env->tick <= start_tick) {
        return 0.0f;
    } else {
        float progress = (env->tick - start_tick) * env->turn_penalty_ramp;
        return env->turn_penalty * progress;
    }
}

void add_log(Artillery3D* env) {
    env->log.perf += env->hit;
    env->log.episode_length += env->tick;
    env->log.score += env->score;
    env->log.dist += env->dist;
    env->log.max_reward_distn += env->max_reward_distn;
    env->log.turn_penaltyn += env->turn_penaltyn;
    env->log.n += 1;
    env->log.acc1000 += 750.0f - env->dist;
    env->log.shots_fired += env->shots_fired;
    env->log.targets_remaining += env->targets_remaining;
}

static inline void compute_observations(Artillery3D* env) {
    env->observations[0] = env->azimuth;
    env->observations[1] = env->elevation;
    env->observations[2] = env->score;
    env->observations[3] = env->tick * 0.01;
    env->observations[4] = env->fuse_t;
    env->observations[5] = env->shots_fired * env->inv_max_shots;
    env->observations[6] = env->targets_remaining / NUMTARGETS;

    int base_obs = 7;
    for (int i = 0; i < NUMTARGETS; i++) {
        int idx = base_obs + i * 6;
        env->observations[idx + 0] = env->tx[i] * env->inv_x_size;
        env->observations[idx + 1] = env->ty[i] * env->inv_y_size;
        env->observations[idx + 2] = env->tz[i] * env->inv_z_size;
        env->observations[idx + 3] = env->target_vx[i] * 0.01;
        env->observations[idx + 4] = env->target_vy[i] * 0.01;
        env->observations[idx + 5] = env->target_vz[i] * 0.01;
    }
}

void reset_round(Artillery3D* env) {
    env->terminals[0] = 0;
    get_random_start(env);
    env->tick = 0;
    env->runs += 1;
    env->score = 0;
    env->fired = 0;
    env->hit = 0.0f;
    env->shots_fired = 0;
    env->shots_remaining = MAXSHOTS;
    env->targets_remaining = NUMTARGETS;
    env->projectile_active = 0;
    env->projectile_time = 0.0f;
    env->fire_t = 99999.0f;
    env->dist = XSIZE;
    env->max_reward_distn = env->max_dist0 - (int)(env->runs * env->dist_fade);
    if (env->max_reward_distn < env->max_reward_dist) env->max_reward_distn = env->max_reward_dist;
    for (int i = 0; i < NUMTARGETS; i++) {
        env->time_target_vanish[i] = 99999999.9f;
    }
}

void c_reset(Artillery3D* env) {
    compute_observations(env);
    reset_round(env);
}

static inline void calculate_distance(Artillery3D* env) {
    env->vx0 = MUZZLEV * cosf(env->azimuth-0.5f);
    env->vy0 = MUZZLEV * sinf(env->azimuth-0.5f);
    env->vz0 = MUZZLEV * sinf(env->elevation);

    float ft = env->fuse_t * FUSEMULT;

    float px = X0 + env->vx0 * ft;
    float py = Y0 + env->vy0 * ft;
    float pz = Z0 + env->vz0 * ft - 0.5f * G * ft * ft;

    env->boom_x = px;
    env->boom_y = py;
    env->boom_z = pz;

    env->dist = XSIZE;
    for (int i = 0; i < NUMTARGETS; i++) {
        if (env->target_vx[i] == 0.0f) continue;

        float txn = env->tx[i] + env->target_vx[i] * ft;
        float tyn = env->ty[i] + env->target_vy[i] * ft;
        float tzn = env->tz[i] + env->target_vz[i] * ft;

        float dx = px - txn;
        float dy = py - tyn;
        float dz = pz - tzn;

        float dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < env->dist) {
            env->dist = dist;
        }
        if (dist < EXPLRAD) {
            env->targets_remaining -= 1;
            env->time_target_vanish[i] = (env->t + ft) * env->render; // Do not wait in headless, wait when rendering
        }
    }
}

static inline void fire_projectile(Artillery3D* env) {
    calculate_distance(env);
    float score;
    env->fire_t = env->t;

    if (env->dist >= env->max_reward_distn) {
        score = env->miss_penalty;
    } else {
        score = env->inv_max_shots * (1.0f - (env->dist / env->max_reward_distn));
    }

    if (env->dist < env->target_size) env->hit = 1.0f;

    env->score += score;
    env->rewards[0] += score;

    if (env->render) {
        env->projectile_active = 1;
        env->projectile_time = 0.0f;
        env->px = X0;
        env->py = Y0;
        env->pz = Z0;
    }

    env->shots_fired += 1;
    env->shots_remaining -= 1;
}

static inline void step_frame(Artillery3D* env, int action) {

    if (!env->projectile_active) {
        if (action == FIRE) {
            env->fired = 1;
            if (env->projectile_active == 0) {
                fire_projectile(env);
            }
        } else if (action == LEFT) {
            if (env->azimuth < 0.99) {
                env->azimuth += 0.01;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == RIGHT) {
            if (env->azimuth > 0.01) {
                env->azimuth -= 0.01;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == UP) {
            if (env->elevation < 0.99) {
                env->elevation += 0.01;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == DOWN) {
            if (env->elevation > 0.01) {
                env->elevation -= 0.01;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == FUSETUP) {
            if (env->fuse_t < 0.99) {
                env->fuse_t += 0.01;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        } else if (action == FUSETDOWN) {
            if (env->fuse_t > 0.01) {
                env->fuse_t -= 0.01;
            }
            else {
                env->score += env->out_bounds_penalty;
            }
        }

        if (action != FIRE) {
            env->turn_penaltyn = get_turn_penalty(env);
            if (env->turn_penaltyn > env->turn_penalty) env->turn_penaltyn = env->turn_penalty;
            env->score += env->turn_penaltyn;
            env->rewards[0] += env->turn_penaltyn;
        }
    }
    else { // Projectile Active
        env->projectile_time += TIMESTEP;
        float pt = env->projectile_time;
        env->px = X0 + env->vx0 * pt;
        env->py = Y0 + env->vy0 * pt;
        env->pz = Z0 + env->vz0 * pt - 0.5f * G * pt * pt;
        if (env->px > XSIZE) {
            env->fired = 0;
            env->projectile_active = 0;
        }
        for (int i = 0; i < NUMTARGETS; i++) {
            if (env->t > env->time_target_vanish[i]) {
                env->tx[i] = 0.0f;
                env->ty[i] = 0.0f;
                env->tz[i] = 0.0f;
                env->target_vx[i] = 0.0f;
                env->target_vy[i] = 0.0f;
                env->target_vz[i] = 0.0f;
            }
        }
    }

    for (int i = 0; i < NUMTARGETS; i++) {
        if (env->tx[i] > 1.0f) {
            env->tx[i] += env->target_vx[i] * TIMESTEP;
            if (env->tx[i] < 0) env->tx[i] = 0.0f;
            if (env->tx[i] > XSIZE) env->tx[i] = XSIZE;

            env->ty[i] += env->target_vy[i] * TIMESTEP;
            if (env->ty[i] < 0) {
                env->ty[i] = 0.0f;
                env->target_vy[i] = env->target_vy[i] * -1.0f;
            }
            if (env->ty[i] > YSIZE) env->ty[i] = YSIZE;

            env->tz[i] += env->target_vz[i] * TIMESTEP;
            if (env->tz[i] < 0) env->tz[i] = 0.0f;
            if (env->tz[i] > ZSIZE) env->tz[i] = ZSIZE;
        }
    }

    if (env->tick < 2 && env->actions[0] == 0) env->score = -1.1f;

    if (
            (env->shots_remaining < 1 && (!env->render || env->px > XSIZE)) ||
            (env->score < -1.0f) ||
            (env->targets_remaining < 1 && (!env->render || env->px > XSIZE))
        ) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
    }
    env->t += TIMESTEP;
}

void c_step(Artillery3D* env) {
    env->terminals[0] = 0;
    env->rewards[0] = 0.0;

    int action = env->actions[0];
    env->tick += 1;
    step_frame(env, action);
    compute_observations(env);
}

void c_close(Artillery3D* env) {
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

    float x = X0 + r * cosf(el) * cosf(az);
    float y = Y0 + r * cosf(el) * sinf(az);
    float z = Z0 + r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){X0, Y0, Z0+30};
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
        client->camera_distance -= wheel * 50.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 1000.0f);
        update_camera_position(client);
    }
}

Client* make_client(Artillery3D* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    InitWindow(1280, 720, "PufferLib Artillery3D");
    SetTargetFPS(30);

    client->camera_distance = 100.0f;
    client->camera_azimuth = -3.14f;
    client->camera_elevation = PI / 5.0f;
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0.0f, 0.0f};

    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    return client;
}

void close_client(Client* client) {
    CloseWindow();
    free(client);
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
    rlSetClipPlanes(1.0, 10000.0);
    BeginDrawing();
    ClearBackground((Color){135, 206, 235, 255});

    BeginMode3D(env->client->camera);

    float corner_size = 20.0f;
    DrawSphere((Vector3){0, 0, 0}, corner_size, YELLOW);
    DrawSphere((Vector3){XSIZE, 0, 0}, corner_size, YELLOW);
    DrawSphere((Vector3){0, YSIZE, 0}, corner_size, YELLOW);
    DrawSphere((Vector3){XSIZE, YSIZE, 0}, corner_size, YELLOW);

    float barrel_length = 40.0f;
    float barrel_radius = 4.0f;

    float azimuth = env->azimuth - 0.5f;
    float elevation = env->elevation;

    Vector3 cannon_pos = { X0, Y0, Z0 };

    Vector3 barrel_direction = {
        cosf(azimuth) * cosf(elevation),
        sinf(azimuth) * cosf(elevation),
        sinf(elevation)
    };

    Vector3 barrel_end = {
        X0 + barrel_length * barrel_direction.x,
        Y0 + barrel_length * barrel_direction.y,
        Z0 + barrel_length * barrel_direction.z
    };

    DrawCylinderEx(cannon_pos, barrel_end, barrel_radius, barrel_radius, 12, DARKGRAY);

    DrawSphere(cannon_pos, 12.0f, GRAY);

    float vx0 = MUZZLEV * barrel_direction.x;
    float vy0 = MUZZLEV * barrel_direction.y;
    float vz0 = MUZZLEV * barrel_direction.z;

    Vector3 prev_point = cannon_pos;
    int segment_count = 0;

    float x;
    float y;
    float z;

    for (float t = TIMESTEP; t < FUSEMULT; t += TIMESTEP) {
        x = X0 + vx0 * t;
        y = Y0 + vy0 * t;
        z = Z0 + vz0 * t - 0.5f * G * t * t;

        if (z < 0 || x < 0 || x > XSIZE || y < 0 || y > YSIZE) break;

        Vector3 current_point = {x, y, z};

        Color trajectory_color = (segment_count % 2 == 0) ? WHITE : LIGHTGRAY;
        DrawLine3D(prev_point, current_point, trajectory_color);

        prev_point = current_point;
        segment_count++;
    }

    if (env->projectile_active) {
        DrawSphere((Vector3){env->px, env->py, env->pz}, 4.0f, BLACK);
        if (env->t > env->fire_t + env->fuse_t * FUSEMULT) {
            DrawSphere((Vector3){env->boom_x, env->boom_y, env->boom_z}, 20.0f, (Color){128, 128, 128, 128});
            if (env->dist < EXPLRAD) {
                DrawSphere((Vector3){env->boom_x, env->boom_y, env->boom_z}, 10.0f+20.0f*(env->t-env->fire_t), (Color){255, 0, 0, 128});
            }
        }
    }

    for (int i = 0; i < NUMTARGETS; i++) {
        if (env->tx[i] > 1.0f) {
            DrawSphere((Vector3){env->tx[i], env->ty[i], env->tz[i]}, 15.0f, BLUE);
        }
    }

    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, WHITE);
    DrawText("Tab: Toggle fullscreen", 10, 50, 16, WHITE);
    DrawText(TextFormat("Score: %.3f", env->score), 10, 70, 20, BLACK);

    EndDrawing();
}
