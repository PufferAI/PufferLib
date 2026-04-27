// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>
#include <stdlib.h>
#include <strings.h>

#include "dronelib.h"

typedef enum {
    IDLE,
    HOVER,
    ORBIT,
    FOLLOW,
    CUBE,
    CONGO,
    FLAG,
    RACE,
    TASK_N // Should always be last
} DroneTask;

static char const* TASK_NAMES[TASK_N] = {"idle", "hover", "orbit", "follow",
                                         "cube", "congo", "flag",  "race"};

DroneTask get_task(char* task_name) {
    for (size_t i = 0; i < TASK_N; i++) {
        if (strcasecmp(TASK_NAMES[i], task_name) == 0) {
            return (DroneTask)i;
        }
    }

    return HOVER;
}

void move_target(Drone* agent) {
    agent->target->pos.x += agent->target->vel.x;
    agent->target->pos.y += agent->target->vel.y;
    agent->target->pos.z += agent->target->vel.z;

    if (agent->target->pos.x < -MARGIN_X || agent->target->pos.x > MARGIN_X) {
        agent->target->vel.x = -agent->target->vel.x;
    }
    if (agent->target->pos.y < -MARGIN_Y || agent->target->pos.y > MARGIN_Y) {
        agent->target->vel.y = -agent->target->vel.y;
    }
    if (agent->target->pos.z < -MARGIN_Z || agent->target->pos.z > MARGIN_Z) {
        agent->target->vel.z = -agent->target->vel.z;
    }
}

void set_target_idle(unsigned int* rng, Drone* agent) {
    agent->target->pos =
        (Vec3){rndf(-MARGIN_X, MARGIN_X, rng), rndf(-MARGIN_Y, MARGIN_Y, rng), rndf(-MARGIN_Z, MARGIN_Z, rng)};
    agent->target->vel =
        (Vec3){rndf(-V_TARGET, V_TARGET, rng), rndf(-V_TARGET, V_TARGET, rng), rndf(-V_TARGET, V_TARGET, rng)};
}

void set_target_hover(unsigned int* rng, Drone* agent, float hover_target_dist) {
    // uniform direction on sphere
    float u = rndf(0.0f, 1.0f, rng);
    float v = rndf(0.0f, 1.0f, rng);
    float z = 2.0f * v - 1.0f;
    float a = 2.0f * (float)M_PI * u;
    float r_xy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    Vec3 dir = (Vec3){r_xy * cosf(a), r_xy * sinf(a), z};

    // uniform radius in ball
    float rad = hover_target_dist * cbrtf(rndf(0.0f, 1.0f, rng));
    Vec3 p = add3(agent->state.pos, scalmul3(dir, rad));

    // clamp to grid bounds
    agent->target->pos = (Vec3){
        clampf(p.x, -MARGIN_X, MARGIN_X),
        clampf(p.y, -MARGIN_Y, MARGIN_Y),
        clampf(p.z, -MARGIN_Z, MARGIN_Z)
    };
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_orbit(Drone* agent, int idx, int num_agents) {
    // Fibbonacci sphere algorithm
    float R = 8.0f;
    float phi = M_PI * (sqrt(5.0f) - 1.0f);
    float y = 1.0f - 2 * ((float)idx / (float)num_agents);
    float radius = sqrtf(1.0f - y * y);

    float theta = phi * idx;
    float x = cos(theta) * radius;
    float z = sin(theta) * radius;

    agent->target->pos = (Vec3){R * x, R * z, R * y}; // convert to z up
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_follow(unsigned int* rng, Drone* agents, int idx) {
    Drone* agent = &agents[idx];

    if (idx == 0) {
        set_target_idle(rng, agent);
    } else {
        agent->target->pos = agents[0].target->pos;
        agent->target->vel = agents[0].target->vel;
    }
}

void set_target_cube(Drone* agent, int idx) {
    float z = idx / 16;
    idx = idx % 16;
    float x = (float)(idx % 4);
    float y = (float)(idx / 4);
    agent->target->pos = (Vec3){4 * x - 6, 4 * y - 6, 4 * z - 6};
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_congo(unsigned int* rng, Drone* agents, int idx) {
    if (idx == 0) {
        set_target_idle(rng, &agents[0]);
        return;
    }

    Drone* follow = &agents[idx - 1];
    Drone* lead = &agents[idx];
    lead->target->pos = follow->target->pos;
    lead->target->vel = follow->target->vel;

    // TODO: Slow hack
    for (int i = 0; i < 40; i++) {
        move_target(lead);
    }
}

void set_target_flag(Drone* agent, int idx) {
    float x = (float)(idx % 8);
    float y = (float)(idx / 8);
    x = 2.0f * x - 7;
    y = 5 - 1.5f * y;
    agent->target->pos = (Vec3){0.0f, x, y};
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_race(Drone* agent) { *agent->target = agent->buffer[agent->buffer_idx]; }

void set_target(unsigned int* rng, DroneTask task, Drone* agents, int idx, int num_agents, float hover_target_dist) {
    Drone* agent = &agents[idx];

    if (task == IDLE) set_target_idle(rng, agent);
    else if (task == HOVER) set_target_hover(rng, agent, hover_target_dist);
    else if (task == ORBIT) set_target_orbit(agent, idx, num_agents);
    else if (task == FOLLOW) set_target_follow(rng, agents, idx);
    else if (task == CUBE) set_target_cube(agent, idx);
    else if (task == CONGO) set_target_congo(rng, agents, idx);
    else if (task == FLAG) set_target_flag(agent, idx);
    else if (task == RACE) set_target_race(agent);
}
