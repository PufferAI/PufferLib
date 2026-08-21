// Ant (gymnasium Ant-v5) on the MuJoCo-style physics core: obs = qpos[2:] +
// qvel + clip(cfrc_ext[1:], -1, 1), reward = healthy + forward velocity - ctrl
// cost - contact cost, terminates when the torso leaves [0.2, 1.0] m. Model:
// resources/mujoco/ant.xml compiled by mjcf2bin.py. mjc_reset/mjc_step are
// host+device (see backend.h).

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
// physics.h capacities sized to this model (checked by mj_loadModel)
#define MJ_MAX_NQ 15
#define MJ_MAX_NV 14
#define MJ_MAX_NBODY 14
#define MJ_MAX_NJNT 9
#define MJ_MAX_NGEOM 14
#define MJ_MAX_NU 8
#define MJ_MAXCON 25
#define MJ_MAXEFC 108
#include "physics.h"
#include "render.h"

#define ANT_FRAME_SKIP 5
#define OBS_SIZE 105
#define NUM_ATNS 8
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1}
#define PUF_STEPS_PER_SEC 20

MjModel mj_model;

struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float x_velocity;
    float distance;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    unsigned int rng;
    const MjModel* m;
    int tick;
    float x_start;
    float episode_return;
    int max_steps;
    float reset_noise_scale;
    float forward_reward_weight;
    float ctrl_cost_weight;
    float contact_cost_weight;
    float healthy_reward;
    MjData d;
};
typedef Env Ant;

// Returns the contact cost (sum of squared clipped external forces)
MJ_HD float compute_observations(Ant* env) {
    const MjModel* m = env->m;
    float* obs = env->agents[0].observations;
    memcpy(obs, env->d.qpos + 2, (m->nq - 2)*sizeof(float));
    memcpy(obs + m->nq - 2, env->d.qvel, m->nv*sizeof(float));
    float* cfrc = obs + m->nq - 2 + m->nv;
    float cost = 0.0f;
    for (int b = 1; b < m->nbody; b++) {
        for (int k = 0; k < 6; k++) {
            float f = fminf(fmaxf(env->d.cfrc_ext[b][k], -1.0f), 1.0f);
            cfrc[6*(b - 1) + k] = f;
            cost += f*f;
        }
    }
    return env->contact_cost_weight*cost;
}

MJ_HD void mjc_reset(Ant* env) {
    const MjModel* m = env->m;
    mj_resetData(m, &env->d);
    float s = env->reset_noise_scale;
    for (int i = 0; i < m->nq; i++) {
        env->d.qpos[i] += s*(2.0f*mju_rand(&env->rng) - 1.0f);
    }
    for (int i = 0; i < m->nv; i++) {
        env->d.qvel[i] = s*mju_randn(&env->rng);
    }
    mj_kinematics(m, &env->d);
    env->tick = 0;
    env->x_start = env->d.qpos[0];
    env->episode_return = 0.0f;
    compute_observations(env);
}

MJ_HD void mjc_step(Ant* env) {
    const MjModel* m = env->m;
    float* actions = env->agents[0].actions;
    float cost = 0.0f;
    for (int i = 0; i < m->nu; i++) {
        env->d.ctrl[i] = fminf(fmaxf(actions[i], -1.0f), 1.0f);
        cost += env->d.ctrl[i]*env->d.ctrl[i];
    }
    // Gym measures the torso displacement with body xpos, which lags qpos by
    // one substep (kinematics of the last forward pass)
    float x0 = env->d.xpos[1][0];
    for (int k = 0; k < ANT_FRAME_SKIP; k++) {
        mj_step(m, &env->d);
    }
    mj_rnePostConstraint(m, &env->d);
    float* q = env->d.qpos;
    float dt = ANT_FRAME_SKIP*m->opt_timestep;
    float x_velocity = (env->d.xpos[1][0] - x0) / dt;
    int healthy = isfinite(q[2]) && q[2] >= 0.2f && q[2] <= 1.0f;
    float contact_cost = compute_observations(env);
    float reward = env->forward_reward_weight*x_velocity - env->ctrl_cost_weight*cost
        - contact_cost + (healthy ? env->healthy_reward : 0.0f);
    env->tick++;
    env->episode_return += reward;
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = 0.0f;
    if (healthy && env->tick < env->max_steps) {
        return;
    }
    float distance = q[0] - env->x_start;
    float xvel = distance / (env->tick*dt);
    env->agents[0].terminals[0] = 1.0f;
    env->log.perf += fminf(fmaxf(xvel / 5.0f, 0.0f), 1.0f);
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.x_velocity += xvel;
    env->log.distance += distance;
    env->log.n += 1.0f;
    mjc_reset(env);
}

void mjc_render(Ant* env) {
    float target[3] = {env->d.xpos[1][0], env->d.xpos[1][1], 0.3f};
    mj_render(env->m, &env->d, "PufferLib Ant", target, 4.0f, 2.0f,
        TextFormat("step %d  x %.2f m  vel %.2f m/s  return %.1f", env->tick,
        env->d.qpos[0], env->d.qvel[0], env->episode_return));
}

void mjc_init(Ant* env, Dict* kwargs) {
    if (mj_model.nbody == 0) {
        mj_loadModel(&mj_model, dict_get_str(kwargs, "model"));
    }
    assert(mj_model.nq - 2 + mj_model.nv + 6*(mj_model.nbody - 1) == OBS_SIZE);
    assert(mj_model.nu == NUM_ATNS);
    env->m = &mj_model;
    env->num_agents = 1;
    env->agents[0].policy = 0;
    env->agents[0].action_mask = NULL;
    env->max_steps = dict_get(kwargs, "max_steps");
    env->reset_noise_scale = dict_get(kwargs, "reset_noise_scale");
    env->forward_reward_weight = dict_get(kwargs, "forward_reward_weight");
    env->ctrl_cost_weight = dict_get(kwargs, "ctrl_cost_weight");
    env->contact_cost_weight = dict_get(kwargs, "contact_cost_weight");
    env->healthy_reward = dict_get(kwargs, "healthy_reward");
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_velocity", log->x_velocity);
    dict_set(out, "distance", log->distance);
}

#include "backend.h"
