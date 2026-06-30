// A sample multiagent coordination env. Star PufferLib on GitHub to support!
// Don't one-line structs/fns/ifs/vars in PRs. This fits in a screenshot.
#include <stdlib.h>
#include <math.h>
#include "raylib.h"

#define AGENTS 8
#define TARGETS 8
const int WIDTH = 1080, HEIGHT = 720, COOLDOWN = 30, TYPES = 4;
const float SPEED = 20.0f, MIN_TICKS = COOLDOWN*AGENTS/(float)TARGETS;
float clip(float val, float min, float max) { return fmaxf(fminf(val, max), min); }

// Required struct. Floats only, n last
typedef struct { float perf, score, n; } Log;
typedef struct { float x, y, heading, speed, type, ticks, cooldown; } Entity;
typedef struct {
    Log log; int num_agents; unsigned int rng; // Required
    float *observations, *actions, *rewards, *terminals; // Required
    Entity entities[AGENTS + TARGETS]; Texture2D sprites;
} Env; // Required: An env struct. You can macro the name in binding.c

void compute_observations(Env* env) {
    int idx = 0; float* obs = env->observations;
    for (int a=0; a<AGENTS ; a++) {
        Entity* agent = &env->entities[a];
        obs[idx++] = agent->heading / (2*PI);
        obs[idx++] = agent->speed / SPEED;
        for (int o=0; o<AGENTS + TARGETS; o++) {
            Entity* other = &env->entities[o];
            obs[idx++] = (other->x - agent->x) / WIDTH;
            obs[idx++] = (other->y - agent->y) / HEIGHT;
            obs[idx++] = other->cooldown / COOLDOWN;
            obs[idx++] = other->type == agent->type ? 1 : 0;
        }
    }
}

void c_reset(Env* env) {
    for (int i=0; i<AGENTS+TARGETS; i++) {
        Entity* entity = &env->entities[i];
        entity->x = 16 + rand_r(&env->rng)%(WIDTH-16);
        entity->y = 16 + rand_r(&env->rng)%(HEIGHT-16);
        entity->type = i % TYPES;
        entity->ticks = 0;
    }
    compute_observations(env);
}

void c_step(Env* env) {
    for (int i=0; i<AGENTS; i++) {
        Entity* agent = &env->entities[i];
        agent->ticks += 1;
        agent->heading += (env->actions[2*i] - 4.0f)/12.0f;
        if (agent->heading < -PI) agent->heading += 2*PI;
        if (agent->heading > PI) agent->heading -= 2*PI;
        float speed = agent->speed;
        agent->speed = clip(speed + (env->actions[2*i + 1] - 2.0f), 0.0f, SPEED);
        agent->x = clip(agent->x + speed*cosf(agent->heading), 16, WIDTH-16);
        agent->y = clip(agent->y + speed*sinf(agent->heading), 16, HEIGHT-16);
        for (int t=0; t<TARGETS; t++) {
            Entity* target = &env->entities[AGENTS + t];
            if (target->cooldown > 0 || target->type != agent->type
                || fabsf(target->x - agent->x) > 32
                || fabsf(target->y - agent->y) > 32) continue;
            target->cooldown = COOLDOWN;
            if (rand_r(&env->rng) % 10 == 0) {
                target->x = 16 + rand_r(&env->rng)%(WIDTH-16);
                target->y = 16 + rand_r(&env->rng)%(HEIGHT-16);
            }
            env->rewards[i] = 1.0f;
            env->log.perf += clip(MIN_TICKS/agent->ticks, 0.0f, 1.0f);
            env->log.score -= agent->ticks;
            env->log.n++;
            agent->type = ((int)agent->type + 1) % TYPES;
            agent->ticks = 0;
            break;
        }
    }
    for (int t=0; t<TARGETS; t++) {
        Entity* target = &env->entities[AGENTS + t];
        target->cooldown = fmaxf(target->cooldown - 1, 0);
    }
    compute_observations(env);
}

void c_render(Env* env) {
    if (!IsWindowReady()) {
        InitWindow(WIDTH, HEIGHT, "PufferLib Env"); SetTargetFPS(30);
        env->sprites = LoadTexture("resources/shared/puffers.png");
    }
    if (IsKeyDown(KEY_ESCAPE)) exit(0);
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (int i=0; i<AGENTS+TARGETS; i++) {
        Entity* entity = &env->entities[i];
        int sz = i < AGENTS ? 32 : 64, y = i < AGENTS ? 576 : 512;
        if (i < AGENTS  && (entity->heading < -PI/2 || entity->heading > PI/2)) y += 32;
        DrawTexturePro(env->sprites,
            (Rectangle){sz*entity->type, y, sz, sz},
            (Rectangle){entity->x - sz/2, entity->y - sz/2, sz, sz},
            (Vector2){0, 0}, 0, entity->cooldown > 0 ? DARKGRAY: WHITE
        );
    }
    EndDrawing();
}

void c_close(Env* env) {
    if (IsWindowReady()) {
        UnloadTexture(env->sprites);
        CloseWindow();
    }
}
