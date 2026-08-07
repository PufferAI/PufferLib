#include "soccer.h"
#include "puffercpu.h"

static void bind_agents(Env* env, obs_t* observations,
        float* actions, float* rewards, float* terminals) {
    for (int i = 0; i < env->num_agents; i++) {
        env->agents[i].observations = observations + i * OBS_SIZE;
        env->agents[i].actions = actions + i * NUM_ATNS;
        env->agents[i].rewards = rewards + i;
        env->agents[i].terminals = terminals + i;
        env->agents[i].action_mask = NULL;
        env->agents[i].policy = i;
    }
}

void demo(void) {
    Weights* weights = load_weights("resources/soccer/soccer_weights.bin");
    int logit_sizes[NUM_ATNS] = ACT_SIZES;
    PufferNet* net = make_puffernet(weights, NUM_TEAMS, OBS_SIZE, 256, 4,
        logit_sizes, NUM_ATNS);

    Env env = {
        .num_agents = NUM_TEAMS,
        .render = 1,
        .rng = 42,
        .max_steps = 3000,
        .frameskip = 1,
        .accel = 120,
        .turn_rate = 2.6f,
        .player_friction = 0.65f,
        .ball_friction = 0.05f,
        .restitution = 0.92f,
        .dt = 0.05f,
        .reward_goal = 0.75f,
        .reward_ball_progress = 0.141525581f,
        .reward_timeout_ball_position = 0.25f,
        .global_agents = 600,
        .timeout_ball_reward_anneal_start = 125000000,
        .timeout_ball_reward_anneal_end = 250000000,
    };
    init(&env);

    obs_t observations[NUM_TEAMS * OBS_SIZE] = {0};
    float actions[NUM_TEAMS * NUM_ATNS] = {0};
    float rewards[NUM_TEAMS] = {0};
    float terminals[NUM_TEAMS] = {0};
    bind_agents(&env, observations, actions, rewards, terminals);
    puf_reset(&env);
    puf_render(&env);

    while (!WindowShouldClose()) {
        forward_puffernet(net, observations, actions);
        // Hold shift to take over red player 0; the policy drives everyone else
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            actions[0] = 1.0f;
            actions[1] = 1.0f;
            if (IsKeyDown(KEY_A)) actions[0] = 0.0f;
            if (IsKeyDown(KEY_D)) actions[0] = 2.0f;
            if (IsKeyDown(KEY_S)) actions[1] = 0.0f;
            if (IsKeyDown(KEY_W)) actions[1] = 2.0f;
        }
        puf_step(&env);
        puf_render(&env);
    }

    free_puffernet(net);
    free(weights);
    puf_close(&env);
    CloseWindow();
}

int main(void) {
    demo();
    return 0;
}
