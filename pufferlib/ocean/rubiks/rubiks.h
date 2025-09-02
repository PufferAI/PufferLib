/* Target: a sample multiagent env about puffers eating stars.
 * Use this as a tutorial and template for your own multiagent envs.
 * We suggest starting with the Squared env for a simpler intro.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
    float score; // Recommended unnormalized single real number perf metric
    float episode_return; // Recommended metric: sum of agent rewards over episode
    float episode_length; // Recommended metric: number of steps of agent episode
    // Any extra fields you add here may be exported to Python in binding.c
    float n; // Required as the last field 
} Log;

typedef struct {
    Texture2D puffer;
    Texture2D star;
} Client;

typedef struct {
    float x;
    float y;
    float heading;
    float speed;
    int ticks_since_reward;
} Agent;

typedef struct {
    float x;
    float y;
} Goal;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Client* client;
    Agent* agents;
    Goal* goals;
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int width;
    int height;
    int num_agents;
    int num_goals;
} Target;

/* Recommended to have an init function of some kind if you allocate 
 * extra memory. This should be freed by c_close. Don't forget to call
 * this in binding.c!
 */
void init(Target* env) {
    env->agents = calloc(env->num_agents, sizeof(Agent));
    env->goals = calloc(env->num_goals, sizeof(Goal));
}

void update_goals(Target* env) {
    for (int a=0; a<env->num_agents; a++) {
        Agent* agent = &env->agents[a];
        for (int g=0; g<env->num_goals; g++) {
            Goal* goal = &env->goals[g];
            float dx = (goal->x - agent->x);
            float dy = (goal->y - agent->y);
            float dist = sqrt(dx*dx + dy*dy);
            if (dist > 32) {
                continue;
            }
            goal->x = rand() % env->width;
            goal->y = rand() % env->height;
            env->rewards[a] = 1.0f;
            env->log.perf += 1.0f;
            env->log.score += 1.0f;
            env->log.episode_length += agent->ticks_since_reward;
            agent->ticks_since_reward = 0;
            env->log.episode_return += 1.0f;
            env->log.n++;
        }
    }
}

/* Recommended to have an observation function of some kind because
 * you need to compute agent observations in both reset and in step.
 * If using float obs, try to normalize to roughly -1 to 1 by dividing
 * by an appropriate constant.
 */
void compute_observations(Target* env) {
    int obs_idx = 0;
    for (int a=0; a<env->num_agents; a++) {
        Agent* agent = &env->agents[a];
        for (int g=0; g<env->num_goals; g++) {
            Goal* goal = &env->goals[g];
            env->observations[obs_idx++] = (goal->x - agent->x)/env->width;
            env->observations[obs_idx++] = (goal->y - agent->y)/env->height;
        }
        for (int a=0; a<env->num_agents; a++) {
            Agent* other = &env->agents[a];
            env->observations[obs_idx++] = (other->x - agent->x)/env->width;
            env->observations[obs_idx++] = (other->y - agent->y)/env->height;
        }
        env->observations[obs_idx++] = agent->heading/(2*PI);
        env->observations[obs_idx++] = env->rewards[a];
        env->observations[obs_idx++] = agent->x/env->width;
        env->observations[obs_idx++] = agent->y/env->height;
    }
}

// Required function
void c_reset(Target* env) {
    for (int i=0; i<env->num_agents; i++) {
        env->agents[i].x = rand() % env->width;
        env->agents[i].y = rand() % env->height;
        env->agents[i].ticks_since_reward = 0;
    }
    for (int i=0; i<env->num_goals; i++) {
        env->goals[i].x = rand() % env->width;
        env->goals[i].y = rand() % env->height;
    }
    compute_observations(env);
}

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

// Required function
void c_step(Target* env) {
    for (int i=0; i<env->num_agents; i++) {
        env->rewards[i] = 0;
        Agent* agent = &env->agents[i];
        agent->ticks_since_reward += 1;

        agent->heading += ((float)env->actions[2*i] - 4.0f)/12.0f;
        agent->heading = clip(agent->heading, 0, 2*PI);

        agent->speed += 1.0f*((float)env->actions[2*i + 1] - 2.0f);
        agent->speed = clip(agent->speed, -20.0f, 20.0f);

        agent->x += agent->speed*cosf(agent->heading);
        agent->x = clip(agent->x, 0, env->width);

        agent->y += agent->speed*sinf(agent->heading);
        agent->y = clip(agent->y, 0, env->height);

        if (agent->ticks_since_reward % 512 == 0) {
            env->agents[i].x = rand() % env->width;
            env->agents[i].y = rand() % env->height;
        }
    }
    update_goals(env);
    compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(Target* env) {
    if (env->client == NULL) {
        InitWindow(env->width, env->height, "PufferLib Target");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));

        // Don't do this before calling InitWindow
        env->client->puffer = LoadTexture("resources/shared/puffers_128.png");
        env->client->star = LoadTexture("resources/target/star.png");
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    for (int i=0; i<env->num_goals; i++) {
        Goal* goal = &env->goals[i];
        DrawTexture(
            env->client->star,
            goal->x - 32,
            goal->y - 32,
            WHITE
        );
    }

    for (int i=0; i<env->num_agents; i++) {
        Agent* agent = &env->agents[i];
        float heading = agent->heading;
        DrawTexturePro(
            env->client->puffer,
            (Rectangle){
                (heading < PI/2 || heading > 3*PI/2) ? 0 : 128,
                0, 128, 128,
            },
            (Rectangle){
                agent->x - 64,
                agent->y - 64,
                128,
                128
            },
            (Vector2){0, 0},
            0,
            WHITE
        );
    }

    EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Target* env) {
    free(env->agents);
    free(env->goals);
    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->puffer);
        UnloadTexture(client->star);
        CloseWindow();
        free(client);
    }
}
