#include <stdio.h>
#include "clifford.h"

int main(void) {
    CliffordVecEnv shared = {0};
    shared.num_envs = 1;
    shared.n_qubits = CLIFFORD_N_QUBITS;
    shared.dim = CLIFFORD_DIM;
    set_difficulty_level(&shared, 10.0);
    shared.max_steps = 200;
    shared.single_qubit_cost = 0.001f;
    shared.cz_cost = 0.1f;
    shared.goal_bonus = 0.0f;
    shared.failure_penalty = -1.0f;
    build_clifford_actions(&shared);
    for (int col = 0; col < shared.dim; ++col) {
        shared.identity_cols[col] = 1ULL << col;
    }

    unsigned char observations[CLIFFORD_OBS_SIZE] = {0};
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    CliffordEnv env = {
        .cols = (uint64_t*)calloc((size_t)shared.dim, sizeof(uint64_t)),
        .vec = &shared,
        .observations = observations,
        .actions = actions,
        .rewards = rewards,
        .terminals = terminals,
        .num_agents = 1,
    };
    rng_seed(&env.rng, 1);
    c_reset(&env);
    for (int step = 0; step < 1000; ++step) {
        actions[0] = (float)sample_action(&shared, &env.rng);
        c_step(&env);
    }

    printf("clifford smoke complete: reward=%f terminal=%f\n", rewards[0], terminals[0]);
    free(env.cols);
    free(shared.actions);
    return 0;
}
