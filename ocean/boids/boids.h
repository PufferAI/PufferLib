#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "raylib.h"

#define BOID_WIDTH 16.0f
#define BOID_HEIGHT 16.0f
#define TOP_MARGIN 50
#define BOTTOM_MARGIN 50
#define LEFT_MARGIN 50
#define RIGHT_MARGIN 50
#define VELOCITY_CAP 5
#define VISUAL_RANGE 400
#define PROTECTED_RANGE ((int)(1.5f * BOID_WIDTH))
#define WIDTH 1080
#define HEIGHT 720
#define BOID_TEXTURE_PATH "./resources/shared/puffers_128.png"
#define EPS 1e-8f // avoids div by zero in angle calc

typedef struct {
    float score;
    float n;
    float t_margin_turn_reward;
    float t_cohesion_reward;
    float t_separation_reward;
    float t_alignment_reward;
} Log;

typedef struct {
    float x;
    float y;
} Velocity;

typedef struct {
    float x;
    float y;
    Velocity velocity;
} Boid;

typedef struct {
    float width;
    float height;
    Texture2D boid_texture;
} Client ;

typedef struct {
    // Flat array of shape (num_agents * 8) values:
    // - Each boid has 8 values corresponding to (x, y, vx, vy, dx, dy, dvx, dvy)
    // - The first 8 values are for the boid itself
    // - All the other 8 values are for the other boids
    float* observations;
    float* actions; // size (num_agents, 2->(dvx, dvy))
    float* rewards; // size (num_agents) with per-boid rewards
    float* terminals;
    Boid* boids;
    unsigned num_agents;
    float margin_turn_factor;
    float cohesion_factor;
    float separation_factor;
    float alignment_factor;
    unsigned tick;
    Log log;
    unsigned report_interval;
    Client* client;
    unsigned rng; // unused but required field for vecenv compatibility
} Boids;

static inline float flmax(float a, float b) { return a > b ? a : b; }
static inline float flmin(float a, float b) { return a > b ? b : a; }
static inline float flclip(float x,float lo,float hi) { return flmin(hi,flmax(lo,x)); }
static inline float rndf(float lo,float hi) { return lo + (float)rand()/(float)RAND_MAX*(hi-lo); }

static void spawn_boid(Boids *env, unsigned int i) {
    env->boids[i].x = rndf(LEFT_MARGIN, WIDTH  - RIGHT_MARGIN);
    env->boids[i].y = rndf(BOTTOM_MARGIN, HEIGHT - TOP_MARGIN);
    env->boids[i].velocity.x = 0;
    env->boids[i].velocity.y = 0;
}

void init(Boids *env) {
    if(env->num_agents < 1) {
        printf("ERROR: num_agents must be bigger than 0\n");
        exit(1);
    }
    if (env->report_interval < 1) {
        printf("ERROR: report_interval must be bigger than 0\n");
        exit(1);
    }
    env->boids = (Boid*)calloc(env->num_agents, sizeof(Boid));
    env->log = (Log){0};
    env->tick = 0;

    for (unsigned idx = 0; idx < env->num_agents; idx++) spawn_boid(env, idx);
}


static void compute_observations(Boids *env) {
    int idx = 0;
    float diff_x, diff_y;
    for (unsigned i=0; i<env->num_agents; i++) {
        // observations for the current boid
        env->observations[idx++] = env->boids[i].x / WIDTH;
        env->observations[idx++] = env->boids[i].y / HEIGHT;
        env->observations[idx++] = env->boids[i].velocity.x / VELOCITY_CAP;
        env->observations[idx++] = env->boids[i].velocity.y / VELOCITY_CAP;
        // zeros for relative observations since comparing to itself will always be 0 (dx, dy, dvx, dvy)
        for (unsigned j=0; j<4; j++) { env->observations[idx++] = 0; }

        // observations for the other boids compared to the current boid
        for (unsigned j=0; j<env->num_agents; j++) {
            if (i == j) continue;
            diff_x = env->boids[i].x - env->boids[j].x;
            diff_y = env->boids[i].y - env->boids[j].y;

            env->observations[idx++] = env->boids[j].x / WIDTH;
            env->observations[idx++] = env->boids[j].y / HEIGHT;
            env->observations[idx++] = env->boids[j].velocity.x / VELOCITY_CAP;
            env->observations[idx++] = env->boids[j].velocity.y / VELOCITY_CAP;
            env->observations[idx++] = diff_x / WIDTH;
            env->observations[idx++] = diff_y / HEIGHT;
            env->observations[idx++] = (env->boids[i].velocity.x - env->boids[j].velocity.x) / VELOCITY_CAP;
            env->observations[idx++] = (env->boids[i].velocity.y - env->boids[j].velocity.y) / VELOCITY_CAP;
        }
    }
}

void apply_action(Boid* boid, float vx, float vy) {
    boid->velocity.x = flclip(boid->velocity.x + vx, -VELOCITY_CAP, VELOCITY_CAP);
    boid->velocity.y = flclip(boid->velocity.y + vy, -VELOCITY_CAP, VELOCITY_CAP);
    boid->x = flclip(boid->x + boid->velocity.x, 0, WIDTH  - BOID_WIDTH);
    boid->y = flclip(boid->y + boid->velocity.y, 0, HEIGHT - BOID_HEIGHT);
}

void c_reset(Boids *env) {
    env->log = (Log){0};
    env->tick = 0;
    for (unsigned boid_indx = 0; boid_indx < env->num_agents; boid_indx++) {
        spawn_boid(env, boid_indx);
    }
    compute_observations(env);
}

void c_step(Boids *env) {
    Boid* current_boid;
    Boid observed_boid;
    float vis_vx_sum, vis_vy_sum, vis_x_sum, vis_y_sum, vis_x_avg, vis_y_avg, vis_vx_avg, vis_vy_avg;
    float diff_x, diff_y, dist, current_boid_reward;
    float margin_turn_reward, cohesion_reward, separation_reward, alignment_reward;
    float protected_x_sum, protected_y_sum;
    float rule_dx, rule_dy, rule_mag;
    unsigned visual_count, protected_count;
    bool manual_control = IsKeyDown(KEY_LEFT_SHIFT);
    float mouse_x = (float)GetMouseX();
    float mouse_y = (float)GetMouseY();

    env->tick++;
    env->rewards[0] = 0.0;
    env->log.score = 0;
    env->log.n = 0;
    env->log.t_margin_turn_reward = 0;
    env->log.t_cohesion_reward = 0;
    env->log.t_separation_reward = 0;
    env->log.t_alignment_reward = 0;
    for (unsigned current_indx = 0; current_indx < env->num_agents; current_indx++) {
        current_boid = &env->boids[current_indx];
        if (manual_control) {
            apply_action(current_boid, (mouse_x - current_boid->x), (mouse_y - current_boid->y));
        } else {
            apply_action(current_boid, (env->actions[current_indx*2] - 1.0f), (env->actions[current_indx*2 + 1] - 1.0f));
        }

        // reward calculation
        current_boid_reward = 0.0f;
        margin_turn_reward = 0.0f;
        cohesion_reward = 0.0f;
        separation_reward = 0.0f;
        alignment_reward = 0.0f;
        protected_count = 0;
        visual_count = 0;
        vis_vx_sum = 0.0f;
        vis_vy_sum = 0.0f;
        vis_x_sum = 0.0f;
        vis_y_sum = 0.0f;
        protected_x_sum = 0.0f;
        protected_y_sum = 0.0f;
        for (unsigned observed_indx = 0; observed_indx < env->num_agents; observed_indx++) {
            if (current_indx == observed_indx) continue;
            observed_boid = env->boids[observed_indx];
            diff_x = current_boid->x - observed_boid.x;
            diff_y = current_boid->y - observed_boid.y;
            dist = sqrtf(diff_x*diff_x + diff_y*diff_y);
            if (dist < PROTECTED_RANGE) {
                protected_count++;
                protected_x_sum += diff_x;
                protected_y_sum += diff_y;
            } else if (dist < VISUAL_RANGE) {
                vis_x_sum += observed_boid.x;
                vis_y_sum += observed_boid.y;
                vis_vx_sum += observed_boid.velocity.x;
                vis_vy_sum += observed_boid.velocity.y;
                visual_count++;
            }
        }
        if (protected_count > 0) {
            rule_mag = sqrtf(protected_x_sum*protected_x_sum + protected_y_sum*protected_y_sum) + EPS;
            separation_reward -= rule_mag * env->separation_factor;
        }
        if (visual_count) {
            vis_x_avg  = vis_x_sum  / visual_count;
            vis_y_avg  = vis_y_sum  / visual_count;
            vis_vx_avg = vis_vx_sum / visual_count;
            vis_vy_avg = vis_vy_sum / visual_count;

            cohesion_reward -= fabsf(vis_x_avg  - current_boid->x) * env->cohesion_factor;
            cohesion_reward -= fabsf(vis_y_avg  - current_boid->y) * env->cohesion_factor;

            rule_dx = vis_vx_avg - current_boid->velocity.x;
            rule_dy = vis_vy_avg - current_boid->velocity.y;
            rule_mag = sqrtf(rule_dx*rule_dx + rule_dy*rule_dy) + EPS;
            alignment_reward -= fabsf(vis_vx_avg - current_boid->velocity.x) * env->alignment_factor;
            alignment_reward -= fabsf(vis_vy_avg - current_boid->velocity.y) * env->alignment_factor;

            rule_dx = vis_x_avg - current_boid->x;
            rule_dy = vis_y_avg - current_boid->y;
            rule_mag = sqrtf(rule_dx*rule_dx + rule_dy*rule_dy) + EPS;
        }

        if (current_boid->y < TOP_MARGIN || current_boid->x < LEFT_MARGIN
            || current_boid->y + BOID_HEIGHT > HEIGHT - BOTTOM_MARGIN
            || current_boid->x + BOID_WIDTH > WIDTH - RIGHT_MARGIN) {
            margin_turn_reward -= env->margin_turn_factor;
        }
        current_boid_reward = margin_turn_reward + cohesion_reward + separation_reward + alignment_reward;

        // Normalization
        // env->rewards[current_indx] = current_boid_reward / 5.0f;
        // env->rewards[current_indx] = current_boid_reward / 205.0f;
        env->rewards[current_indx] = current_boid_reward / 64.0f;

        //log updates
        if (env->tick == env->report_interval) {
            env->log.score              += env->rewards[current_indx];
            env->log.t_margin_turn_reward += margin_turn_reward;
            env->log.t_cohesion_reward    += cohesion_reward;
            env->log.t_separation_reward  += separation_reward;
            env->log.t_alignment_reward   += alignment_reward;
            env->log.n                  += 1.0f;
        }
    }

    if (env->tick == env->report_interval) env->tick = 0;
    compute_observations(env);
}

void c_close_client(Client* client) {
    UnloadTexture(client->boid_texture);
    CloseWindow();
    free(client);
}

void c_close(Boids* env) {
    free(env->boids);
    if (env->client != NULL) {
        c_close_client(env->client);
    }
}

Client* make_client(Boids* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    InitWindow(WIDTH, HEIGHT, "PufferLib Boids");
    SetTargetFPS(60);

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->boid_texture = LoadTexture(BOID_TEXTURE_PATH);
    if (client->boid_texture.id == 0) {
        TraceLog(LOG_ERROR, "Failed to load texture: %s", BOID_TEXTURE_PATH);
        c_close_client(client);
        return NULL;
    }

    return client;
}

void c_render(Boids* env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }
    if (WindowShouldClose() || !IsWindowReady()) {
        TraceLog(LOG_WARNING, "Window is not ready or should close");
        return;
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (unsigned boid_indx = 0; boid_indx < env->num_agents; boid_indx++) {
        DrawTexturePro(
            env->client->boid_texture,
            (Rectangle){
                (env->boids[boid_indx].velocity.x > 0) ? 0.0f : 128.0f,
                0.0f,
                128.0f,
                128.0f,
            },
            (Rectangle){
                env->boids[boid_indx].x,
                env->boids[boid_indx].y,
                BOID_WIDTH,
                BOID_HEIGHT
            },
            (Vector2){0.0f, 0.0f},
            0,
            WHITE
        );
    }
    EndDrawing();
}
