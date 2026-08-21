// Humanoid (gymnasium Humanoid-v5) on the MuJoCo-style physics core: obs =
// qpos[2:] + qvel + cinert[1:] + cvel[1:] + qfrc_actuator[6:] + cfrc_ext[1:],
// reward = healthy + forward COM velocity - ctrl cost - contact cost (clamped
// at 10), terminates when the torso leaves z in (1, 2). Model:
// resources/mujoco/humanoid.xml compiled by mjcf2bin.py. mjc_reset/mjc_step
// are host+device (see backend.h).

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"
// physics.h capacities sized to this model (checked by mj_loadModel)
#define MJ_MAX_NQ 24
#define MJ_MAX_NV 23
#define MJ_MAX_NBODY 14
#define MJ_MAX_NJNT 18
#define MJ_MAX_NGEOM 18
#define MJ_MAX_NU 17
#define MJ_MAXCON 32
#define MJ_MAXEFC 145
#include "physics.h"
#include "render.h"

#define HM_FRAME_SKIP 5
#define HM_CONTACT_COST_MAX 10.0f
#define OBS_SIZE 348
#define NUM_ATNS 17
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
#define PUF_STEPS_PER_SEC 67

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
typedef Env Humanoid;

// Returns the contact cost (sum of squared external forces, clamped)
MJ_HD float compute_observations(Humanoid* env) {
    const MjModel* m = env->m;
    float* obs = env->agents[0].observations;
    memcpy(obs, env->d.qpos + 2, (m->nq - 2)*sizeof(float));
    obs += m->nq - 2;
    memcpy(obs, env->d.qvel, m->nv*sizeof(float));
    obs += m->nv;
    memcpy(obs, env->d.cinert[1], 10*(m->nbody - 1)*sizeof(float));
    obs += 10*(m->nbody - 1);
    memcpy(obs, env->d.cvel[1], 6*(m->nbody - 1)*sizeof(float));
    obs += 6*(m->nbody - 1);
    memcpy(obs, env->d.qfrc_actuator + 6, (m->nv - 6)*sizeof(float));
    obs += m->nv - 6;
    memcpy(obs, env->d.cfrc_ext[1], 6*(m->nbody - 1)*sizeof(float));
    float cost = 0.0f;
    for (int b = 1; b < m->nbody; b++) {
        cost += mju_dot6(env->d.cfrc_ext[b], env->d.cfrc_ext[b]);
    }
    return fminf(env->contact_cost_weight*cost, HM_CONTACT_COST_MAX);
}

// x of the whole-body center of mass from the inertial frames of the last
// kinematics pass (Gym's mass_center)
MJ_HD float mjc_com_x(Humanoid* env) {
    const MjModel* m = env->m;
    float num = 0.0f;
    float den = 0.0f;
    for (int b = 0; b < m->nbody; b++) {
        num += m->body_mass[b]*env->d.xipos[b][0];
        den += m->body_mass[b];
    }
    return num / den;
}

MJ_HD void mjc_reset(Humanoid* env) {
    const MjModel* m = env->m;
    mj_resetData(m, &env->d);
    float s = env->reset_noise_scale;
    for (int i = 0; i < m->nq; i++) {
        env->d.qpos[i] += s*(2.0f*mju_rand(&env->rng) - 1.0f);
    }
    for (int i = 0; i < m->nv; i++) {
        env->d.qvel[i] = s*(2.0f*mju_rand(&env->rng) - 1.0f);
    }
    mj_forward(m, &env->d);
    env->tick = 0;
    env->x_start = env->d.qpos[0];
    env->episode_return = 0.0f;
    compute_observations(env);
}

MJ_HD void mjc_step(Humanoid* env) {
    const MjModel* m = env->m;
    float* actions = env->agents[0].actions;
    float cost = 0.0f;
    for (int i = 0; i < m->nu; i++) {
        const float* range = m->actuator_ctrlrange[i];
        env->d.ctrl[i] = fminf(fmaxf(actions[i], range[0]), range[1]);
        cost += env->d.ctrl[i]*env->d.ctrl[i];
    }
    float x0 = mjc_com_x(env);
    for (int k = 0; k < HM_FRAME_SKIP; k++) {
        mj_step(m, &env->d);
    }
    mj_rnePostConstraint(m, &env->d);
    float* q = env->d.qpos;
    float dt = HM_FRAME_SKIP*m->opt_timestep;
    float x_velocity = (mjc_com_x(env) - x0) / dt;
    int healthy = q[2] > 1.0f && q[2] < 2.0f;
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
    env->log.perf += fminf(fmaxf(xvel / 3.0f, 0.0f), 1.0f);
    env->log.score += env->episode_return;
    env->log.episode_return += env->episode_return;
    env->log.episode_length += env->tick;
    env->log.x_velocity += xvel;
    env->log.distance += distance;
    env->log.n += 1.0f;
    mjc_reset(env);
}

void mjc_render(Humanoid* env) {
    float target[3] = {env->d.xpos[1][0], env->d.xpos[1][1], 1.0f};
    mj_render(env->m, &env->d, "PufferLib Humanoid", target, 5.0f, 1.5f,
        TextFormat("step %d  x %.2f m  vel %.2f m/s  return %.1f", env->tick,
        env->d.qpos[0], env->d.qvel[0], env->episode_return));
}

void mjc_init(Humanoid* env, Dict* kwargs) {
    if (mj_model.nbody == 0) {
        mj_loadModel(&mj_model, dict_get_str(kwargs, "model"));
    }
    assert(mj_model.nq - 2 + 2*mj_model.nv - 6 + 22*(mj_model.nbody - 1) == OBS_SIZE);
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
