// Parity check of an ocean/mujoco env against a MuJoCo (gymnasium) reference
// trajectory written by tests/mujoco_parity.py. Build with
// -DENV_HEADER='"../ocean/mujoco/mjc_ENV.h"' and the env's reward weights
// (-DCTRL_COST_WEIGHT=..., -DHEALTHY_REWARD=..., -DCONTACT_COST_WEIGHT=...,
// -DFORWARD_REWARD_WEIGHT=...).
// Usage: test_mujoco_parity REF_FILE MODEL_BIN
#include <stdio.h>
#include <time.h>
#include ENV_HEADER

#define MAXT 4096
#ifndef FORWARD_REWARD_WEIGHT
#define FORWARD_REWARD_WEIGHT 1.0f
#endif

double ref_q0[MAXT][MJ_MAX_NQ], ref_v0[MAXT][MJ_MAX_NV], ref_a[MAXT][MJ_MAX_NU];
double ref_q[MAXT][MJ_MAX_NQ], ref_v[MAXT][MJ_MAX_NV];
double ref_r[MAXT], ref_obs[MAXT][OBS_SIZE];
int ref_ncon[MAXT], ref_term[MAXT];

void set_state(Env* env, int t) {
    for (int i = 0; i < mj_model.nq; i++) env->d.qpos[i] = ref_q0[t][i];
    for (int i = 0; i < mj_model.nv; i++) env->d.qvel[i] = ref_v0[t][i];
    mj_kinematics(&mj_model, &env->d);
    env->tick = 0;
}

int main(int argc, char** argv) {
    mj_loadModel(&mj_model, argv[2]);
    FILE* fp = fopen(argv[1], "r");
    int T, nq, nv, nu, nobs;
    fscanf(fp, "%d %d %d %d %d", &T, &nq, &nv, &nu, &nobs);
    if (nq != mj_model.nq || nv != mj_model.nv || nu != mj_model.nu || nobs != OBS_SIZE) {
        printf("size mismatch: ref nq %d nv %d nu %d nobs %d vs model %d %d %d %d\n",
            nq, nv, nu, nobs, mj_model.nq, mj_model.nv, mj_model.nu, OBS_SIZE);
        return 1;
    }
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < mj_model.nq; i++) fscanf(fp, "%lf", &ref_q0[t][i]);
        for (int i = 0; i < mj_model.nv; i++) fscanf(fp, "%lf", &ref_v0[t][i]);
        for (int i = 0; i < mj_model.nu; i++) fscanf(fp, "%lf", &ref_a[t][i]);
        for (int i = 0; i < mj_model.nq; i++) fscanf(fp, "%lf", &ref_q[t][i]);
        for (int i = 0; i < mj_model.nv; i++) fscanf(fp, "%lf", &ref_v[t][i]);
        fscanf(fp, "%lf %d %d", &ref_r[t], &ref_ncon[t], &ref_term[t]);
        for (int i = 0; i < OBS_SIZE; i++) fscanf(fp, "%lf", &ref_obs[t][i]);
    }
    fclose(fp);

    float obs[OBS_SIZE], actions[MJ_MAX_NU], reward, terminal;
    Env env = {0};
    env.m = &mj_model;
    mj_makeData(&env.d, (float*)calloc(MJ_SCRATCH, sizeof(float)), 1);
    env.agents[0].observations = obs;
    env.agents[0].actions = actions;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    env.num_agents = 1;
    env.max_steps = 1 << 30;
    env.reset_noise_scale = 0.1f;
    env.forward_reward_weight = FORWARD_REWARD_WEIGHT;
    env.ctrl_cost_weight = CTRL_COST_WEIGHT;
#ifdef HEALTHY_REWARD
    env.healthy_reward = HEALTHY_REWARD;
#endif
#ifdef CONTACT_COST_WEIGHT
    env.contact_cost_weight = CONTACT_COST_WEIGHT;
#endif
    Env env0 = env;
    puf_reset(&env);

    // One-step parity: start every step from the reference start state. The env
    // resets itself on termination, so terminal steps only compare the flag.
    double max_q = 0, max_v = 0, max_r = 0, sum_q = 0, sum_v = 0;
    int arg_q = -1, arg_v = -1, dof_q = -1, dof_v = -1, ncon_mismatch = 0, term_mismatch = 0;
    int arg_r = -1;
    int ncmp = 0;
    for (int t = 0; t < T; t++) {
        set_state(&env, t);
        for (int i = 0; i < mj_model.nu; i++) actions[i] = ref_a[t][i];
        double ret0 = env.episode_return;
        puf_step(&env);
        // the trainer reward is scaled; episode_return accumulates the Gym reward
        double gym_reward = env.episode_return - ret0;
        term_mismatch += (terminal != 0) != (ref_term[t] != 0);
        if (ref_term[t] || terminal) {
            continue;
        }
        ncmp++;
        for (int i = 0; i < mj_model.nq; i++) {
            double eq = fabs(env.d.qpos[i] - ref_q[t][i]);
            sum_q += eq;
            if (eq > max_q) { max_q = eq; arg_q = t; dof_q = i; }
        }
        for (int i = 0; i < mj_model.nv; i++) {
            double ev = fabs(env.d.qvel[i] - ref_v[t][i]);
            sum_v += ev;
            if (ev > max_v) { max_v = ev; arg_v = t; dof_v = i; }
        }
        double er = fabs(gym_reward - ref_r[t]);
        if (er > max_r) { max_r = er; arg_r = t; }
        if (env.d.ncon != ref_ncon[t]) ncon_mismatch++;
    }
    printf("one-step: max|dq| %.3e (t=%d i=%d) max|dv| %.3e (t=%d i=%d)\n",
        max_q, arg_q, dof_q, max_v, arg_v, dof_v);
    printf("one-step: mean|dq| %.3e mean|dv| %.3e max|dr| %.3e (t=%d) ncon mismatch"
        " %d/%d terminal mismatch %d\n", sum_q / (ncmp*mj_model.nq), sum_v / (ncmp*mj_model.nv),
        max_r, arg_r, ncon_mismatch, ncmp, term_mismatch);

    // Free run from the initial state until the reference episode ends. Reward
    // is compared while the trajectories are still in sync.
    set_state(&env, 0);
    double ret = 0, ref_ret = 0, sync_dr = 0;
    int nsync = 0;
    printf("free-run t | max|dq| | max|dv| | qpos[0] vs ref\n");
    for (int t = 0; t < T; t++) {
        if (ref_term[t]) {
            printf("reference episode terminated at step %d, stopping free run\n", t + 1);
            break;
        }
        for (int i = 0; i < mj_model.nu; i++) actions[i] = ref_a[t][i];
        double ret0 = env.episode_return;
        puf_step(&env);
        double gym_reward = env.episode_return - ret0;
        ret += gym_reward;
        ref_ret += ref_r[t];
        double eq = 0, ev = 0;
        for (int i = 0; i < mj_model.nq; i++) eq = fmax(eq, fabs(env.d.qpos[i] - ref_q[t][i]));
        for (int i = 0; i < mj_model.nv; i++) ev = fmax(ev, fabs(env.d.qvel[i] - ref_v[t][i]));
        if (eq < 1e-4) {
            sync_dr = fmax(sync_dr, fabs(gym_reward - ref_r[t]));
            nsync++;
        }
        int n = t + 1;
        if (n == 1 || n == 2 || n == 5 || n == 10 || n == 20 || n == 50 || n == 100 || n == 200
                || n == T) {
            printf("%9d | %.3e | %.3e | %.4f vs %.4f\n", n, eq, ev, env.d.qpos[0], ref_q[t][0]);
        }
    }
    printf("free-run return %.3f vs ref %.3f; max|dr| %.3e over %d in-sync steps\n", ret, ref_ret,
        sync_dr, nsync);

    // Throughput with random actions
    env = env0;
    env.max_steps = 1000;
    puf_reset(&env);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int N = 20000;
    int bad = 0;
    for (int t = 0; t < N; t++) {
        for (int i = 0; i < mj_model.nu; i++) actions[i] = 2.0f*((float)rand() / RAND_MAX) - 1.0f;
        puf_step(&env);
        bad += !isfinite(env.d.qpos[0]) || !isfinite(reward);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9*(t1.tv_nsec - t0.tv_nsec);
    printf("random policy: %.0f steps/s (single thread), %d episodes, mean return %.1f, "
        "nonfinite %d\n", N / dt, (int)env.log.n, env.log.n > 0 ? env.log.score / env.log.n : 0.0f,
        bad);
    return 0;
}
