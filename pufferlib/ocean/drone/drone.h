// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/tensaur/drone

#pragma once

#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

#include "dronelib.h"
#include "tasks.h"

typedef struct Client Client;
typedef struct DroneEnv DroneEnv;

struct DroneEnv {
  float *observations;
  float *actions;
  float *rewards;
  unsigned char *terminals;

  Log log;
  int tick;
  int report_interval;

  DroneTask task;
  int num_agents;
  Drone *agents;

  int max_rings;
  Target* ring_buffer;

  Client *client;
};

void init(DroneEnv *env) {
  env->agents = (Drone*) calloc(env->num_agents, sizeof(Drone));
  env->ring_buffer = (Target*) calloc(env->max_rings, sizeof(Target));

  for (int i = 0; i < env->num_agents; i++) {
    env->agents[i].target = (Target*) calloc(1, sizeof(Target));
  }

  env->log = (Log){0};
  env->tick = 0;
}

void add_log(DroneEnv *env, int idx, bool oob, bool ring_collision,
             bool timeout) {
  Drone *agent = &env->agents[idx];
  env->log.score += agent->score;
  env->log.episode_return += agent->episode_return;
  env->log.episode_length += agent->episode_length;
  env->log.collision_rate += agent->collisions / (float)agent->episode_length;
  env->log.perf += agent->score / (float)agent->episode_length;
  if (oob) {
    env->log.oob += 1.0f;
  }
  if (ring_collision) {
    env->log.ring_collision += 1.0f;
  }
  if (timeout) {
    env->log.timeout += 1.0f;
  }
  env->log.n += 1.0f;

  agent->episode_length = 0;
  agent->episode_return = 0.0f;
}

void compute_observations(DroneEnv *env) {
  int idx = 0;

  for (int i = 0; i < env->num_agents; i++) {
    Drone *agent = &env->agents[i];

    Quat q_inv = quat_inverse(agent->state.quat);
    Vec3 linear_vel_body = quat_rotate(q_inv, agent->state.vel);
    Vec3 to_target = sub3(agent->target->pos, agent->state.pos);

    env->observations[idx++] = linear_vel_body.x / agent->params.max_vel;
    env->observations[idx++] = linear_vel_body.y / agent->params.max_vel;
    env->observations[idx++] = linear_vel_body.z / agent->params.max_vel;

    env->observations[idx++] = agent->state.omega.x / agent->params.max_omega;
    env->observations[idx++] = agent->state.omega.y / agent->params.max_omega;
    env->observations[idx++] = agent->state.omega.z / agent->params.max_omega;

    env->observations[idx++] = agent->state.quat.w;
    env->observations[idx++] = agent->state.quat.x;
    env->observations[idx++] = agent->state.quat.y;
    env->observations[idx++] = agent->state.quat.z;

    env->observations[idx++] = agent->state.rpms[0] / agent->params.max_rpm;
    env->observations[idx++] = agent->state.rpms[1] / agent->params.max_rpm;
    env->observations[idx++] = agent->state.rpms[2] / agent->params.max_rpm;
    env->observations[idx++] = agent->state.rpms[3] / agent->params.max_rpm;

    env->observations[idx++] = to_target.x / GRID_X;
    env->observations[idx++] = to_target.y / GRID_Y;
    env->observations[idx++] = to_target.z / GRID_Z;

    env->observations[idx++] = clampf(to_target.x, -1.0f, 1.0f);
    env->observations[idx++] = clampf(to_target.y, -1.0f, 1.0f);
    env->observations[idx++] = clampf(to_target.z, -1.0f, 1.0f);

    env->observations[idx++] = agent->target->normal.x;
    env->observations[idx++] = agent->target->normal.y;
    env->observations[idx++] = agent->target->normal.z;

    // Multiagent obs
    Drone *nearest = nearest_drone(agent, env->agents, env->num_agents);
    if (env->num_agents > 1) {
      env->observations[idx++] =
          clampf(nearest->state.pos.x - agent->state.pos.x, -1.0f, 1.0f);
      env->observations[idx++] =
          clampf(nearest->state.pos.y - agent->state.pos.y, -1.0f, 1.0f);
      env->observations[idx++] =
          clampf(nearest->state.pos.z - agent->state.pos.z, -1.0f, 1.0f);
    } else {
      env->observations[idx++] = MAX_DIST;
      env->observations[idx++] = MAX_DIST;
      env->observations[idx++] = MAX_DIST;
    }
  }
}

void reset_agent(DroneEnv *env, Drone *agent, int idx) {
  agent->last_dist_reward = 0.0f;
  agent->episode_return = 0.0f;
  agent->episode_length = 0;
  agent->collisions = 0.0f;
  agent->score = 0.0f;

  agent->buffer = env->ring_buffer;
  agent->buffer_size = env->max_rings;
  agent->buffer_idx = -1;

  float size = rndf(0.1f, 0.4f);
  init_drone(agent, size, 0.1f);

  agent->state.pos =
      (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y),
             rndf(-MARGIN_Z, MARGIN_Z)};

  if (env->task == RACE) {
    while (norm3(sub3(agent->state.pos, env->ring_buffer[0].pos)) <
           2.0f * RING_RADIUS) {
      agent->state.pos =
          (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y),
                 rndf(-MARGIN_Z, MARGIN_Z)};
    }
  }

  agent->prev_pos = agent->state.pos;
}

void c_reset(DroneEnv *env) {
  env->tick = 0;
  int rng = rand();

  if (rng > INT_MAX / 2) {
    env->task = RACE;
  } else {
    env->task = (DroneTask)(rng % (TASK_N - 1));
  }

  if (env->task == RACE) {
    reset_rings(env->ring_buffer, env->max_rings);
  }

  for (int i = 0; i < env->num_agents; i++) {
    Drone *agent = &env->agents[i];
    reset_agent(env, agent, i);
    set_target(env->task, env->agents, i, env->num_agents);
  }

  compute_observations(env);
}

void c_step(DroneEnv *env) {
  env->tick = (env->tick + 1) % HORIZON;

  for (int i = 0; i < env->num_agents; i++) {
    Drone *agent = &env->agents[i];
    env->rewards[i] = 0;
    env->terminals[i] = 0;

    float *atn = &env->actions[4 * i];
    move_drone(agent, atn);

    bool out_of_bounds =
        agent->state.pos.x < -GRID_X || agent->state.pos.x > GRID_X ||
        agent->state.pos.y < -GRID_Y || agent->state.pos.y > GRID_Y ||
        agent->state.pos.z < -GRID_Z || agent->state.pos.z > GRID_Z;

    move_target(agent);

    bool collision = check_collision(agent, env->agents, env->num_agents);
    float reward = 0.0f;

    if (env->task == RACE) {
      // Check ring passage
      Target *ring = &env->ring_buffer[agent->buffer_idx];
      int ring_passage = check_ring(agent, ring);

      // Ring collision
      if (ring_passage < 0) {
        env->rewards[i] = (float)ring_passage;
        agent->episode_return += (float)ring_passage;
        env->terminals[i] = 1;
        add_log(env, i, false, true, false);
        reset_agent(env, agent, i);
        set_target(env->task, env->agents, i, env->num_agents);
        continue;
      }

      // Successfully passed through ring - advance to next
      if (ring_passage > 0) {
        set_target(env->task, env->agents, i, env->num_agents);
        env->log.rings_passed += 1.0f;
      }

      reward = dynamic_task_reward(agent, collision, ring_passage);
    } else {
      reward = static_task_reward(agent, collision);
    }

    // Update agent state
    agent->episode_length++;
    agent->score += reward;
    if (collision) {
      agent->collisions += 1.0f;
    }

    env->rewards[i] = reward;
    agent->episode_return += reward;

    // Check termination conditions
    if (out_of_bounds) {
      env->rewards[i] -= 1.0f;
      agent->episode_return -= 1.0f;
      env->terminals[i] = 1;
      add_log(env, i, true, false, false);

      reset_agent(env, agent, i);
      set_target(env->task, env->agents, i, env->num_agents);
      static_task_reward(agent, false);
    } else if (env->tick >= HORIZON - 1) {
      env->terminals[i] = 1;
      add_log(env, i, false, false, true);
    }
  }

  if (env->tick >= HORIZON - 1) {
    c_reset(env);
  }

  compute_observations(env);
}

void c_close_client(Client* client);

void c_close(DroneEnv *env) {
  for (int i = 0; i < env->num_agents; i++) {
    free(env->agents[i].target);
  }

  free(env->agents);
  free(env->ring_buffer);

  if (env->client != NULL) {
    c_close_client(env->client);
  }
}

