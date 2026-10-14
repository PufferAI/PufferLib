/* A dependency-free CPU bridge except for the upstream API's header files.
 * Used for independent mechanics/controller tests and cross-engine trials.
 */
#define NP_HEADLESS 1
#ifdef NP_TEST_ENV_API
#include "../ocean/npole/npole.h"
#endif
#include "../ocean/npole/npole_controller.h"

static NpModel *test_model;

int np_test_load(const char *path) {
    free(test_model);
    test_model = np_load_model(path);
    return test_model ? 0 : 1;
}

int np_test_acceleration(const double *state, double action, double *out) {
    return np_acceleration(state, action, out);
}

int np_test_integrate(double *state, double action, int substeps, double dt) {
    return np_integrate(state, action, substeps, dt);
}

int np_test_encode(const double *state, double last, double *out) {
    return test_model ? np_encode(test_model, state, last, out) : 1;
}

int np_test_inverse(const double *latent, double last, int tick,
        double *estimate, double *diagnostics) {
    if (!test_model) {
        return 1;
    }
    NpPolicy policy = {0};
    policy.tick = tick;
    memcpy(policy.latent, latent, sizeof(policy.latent));
    int result = np_estimate(test_model, &policy, last);
    memcpy(estimate, policy.estimate, sizeof(policy.estimate));
    diagnostics[0] = policy.residual;
    diagnostics[1] = policy.condition;
    return result;
}

int np_test_act(const double *state, double last, double pending, int tick,
        double *out, double *estimate) {
    if (!test_model) {
        return 1;
    }
    NpState physical = {0};
    NpPolicy policy = {0};
    memcpy(physical.y, state, sizeof(physical.y));
    physical.last_applied = last;
    physical.pending = pending;
    policy.tick = tick;
    int result = np_policy_act(test_model, &policy, &physical, out);
    memcpy(estimate, policy.estimate, sizeof(policy.estimate));
    return result;
}

int np_test_policy(const double *state, double last, double pending, int tick,
        double *out, double *estimate, double *diagnostics) {
    if (!test_model) {
        return 1;
    }
    NpState physical = {0};
    NpPolicy policy = {0};
    memcpy(physical.y, state, sizeof(physical.y));
    physical.last_applied = last;
    physical.pending = pending;
    policy.tick = tick;
    int result = np_policy_act(test_model, &policy, &physical, out);
    memcpy(estimate, policy.estimate, sizeof(policy.estimate));
    diagnostics[0] = policy.residual;
    diagnostics[1] = policy.condition;
    return result;
}

int np_test_step(double *y, double command, double *last, double *pending) {
    NpState state = {0};
    memcpy(state.y, y, sizeof(state.y));
    state.last_applied = *last;
    state.pending = *pending;
    int result = np_tick(&state, command);
    memcpy(y, state.y, sizeof(state.y));
    *last = state.last_applied;
    *pending = state.pending;
    return result;
}

void np_test_reset(uint32_t seed, double *out) {
    NpEpisode episode;
    np_reset_seed(&episode, seed);
    memcpy(out, episode.state.y, sizeof(episode.state.y));
}

int np_test_goal(const double *state) {
    return np_good(state);
}

int np_test_api(void) {
#ifdef NP_TEST_ENV_API
    Env env = {0};
    Dict kwargs = {0};
    float observations[OBS_SIZE] = {0.0f};
    float action = 0.7f;
    float reward = 0.0f;
    float terminal = 0.0f;
    env.agents[0].observations = observations;
    env.agents[0].actions = &action;
    env.agents[0].rewards = &reward;
    env.agents[0].terminals = &terminal;
    puf_init(&env, &kwargs);
    puf_reset(&env);
    double zero[NP_STATE] = {0.0};
    np_reset(&env.episode, zero);
    puf_step(&env);
    if (env.num_agents != 1 || env.episode.state.tick != 1
        || env.episode.state.y[NP_DOF] != 0.0 || observations[32] != 0.0f
        || observations[33] != action || terminal != 0.0f) {
        return 1;
    }
    action = -0.2f;
    puf_step(&env);
    if (env.episode.state.y[NP_DOF] <= 0.0 || observations[32] != 0.7f
        || observations[33] != action) {
        return 2;
    }
    action = NAN;
    puf_step(&env);
    if (terminal != 1.0f || env.log.n != 1.0f || env.log.numeric_failure != 1.0f
        || env.log.perf != 0.0f || env.episode.state.tick != 0) {
        return 3;
    }
    return 0;
#else
    return -1;
#endif
}
