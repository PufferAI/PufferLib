#pragma once

// Environment contract consumed by PuffeRL. No trainer types are needed here;
// dependent policy-layout members are checked when the trainer instantiates it.
template<class Vec>
static bool racing_policy_layout(Vec* vec) {
#ifdef RACING_MULTI
    assert((vec->num_policies == 1 || vec->num_policies == racing_race.cars)
        && "racing_multi needs one shared policy or one policy per car slot");
    for (int i = 0; i <= vec->num_policies; i++)
        vec->policy_layout[i] = i * (vec->total_agents / vec->num_policies);
    return true;
#else
    if (!racing_ghosts.count) return false;
    assert(vec->num_policies == vec->total_agents && vec->total_agents == racing_ghosts.count);
    for (int i = 0; i <= vec->num_policies; i++) vec->policy_layout[i] = i;
    return true;
#endif
}
#define PUF_ENV_POLICY_LAYOUT racing_policy_layout
#define PUF_ENV_COMMAND_HEADER "../ocean/racing/trainer_commands.cuh"
#define PUF_ENV_COMMAND racing_trainer_command

#ifdef RACING_MULTI
// Mask is sampled before acting: a finishing/crashing transition remains valid.
#define PUF_ENV_TRAINING_MASK(env) (!(env).task.done)
#define PUF_ENV_TRAINING_ROWS(vec) ((vec)->policy_layout[1])
#define PUF_ENV_LOGGED_AGENTS(vec) ((vec)->policy_layout[1])
#define PUF_ENV_INDEPENDENT_POLICIES true
#define PUF_ENV_EXTERNAL_STEP
#define PUF_ENV_LOAD_INITIAL_POLICY
#define PUF_ENV_RESET_OPPONENT_STATE
#define PUF_ENV_CONFIGURE_TRAINING(kwargs, gamma) dict_set((kwargs), "reward_discount", (gamma))

struct RacingSelfplayState {
    long promotions, promotion_games;
    float promotion_score;
    char challenger[4096];
};
#define PUF_ENV_SELFPLAY_STATE RacingSelfplayState
#define PUF_ENV_TRAINER_HEADER "../ocean/racing/trainer_callbacks.cuh"
#define PUF_ENV_CONFIGURE_EVAL racing_configure_eval
#define PUF_ENV_SELFPLAY_INIT racing_selfplay_init
#define PUF_ENV_SELFPLAY_CHECKPOINT racing_selfplay_promote
#define PUF_ENV_SELFPLAY_PINNED_POLICY(index) ((index) <= 2)
#define PUF_ENV_SELFPLAY_LOG racing_selfplay_log
#define PUF_ENV_VALIDATE_TRAINING racing_validate_training
#else
#define PUF_ENV_INDEPENDENT_POLICIES (racing_ghosts.count > 0)
#endif
