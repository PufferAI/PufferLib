#include "tower_defence.h"
#include "puffernet.h"
#include <unistd.h>

#ifndef TD_DEMO_WEIGHT_PATH
#define TD_DEMO_WEIGHT_PATH "resources/tower_defence/tower_defence_weights.bin"
#endif

#ifndef TD_DEMO_MAX_EPISODE_STEPS
#define TD_DEMO_MAX_EPISODE_STEPS TD_DEFAULT_MAX_EPISODE_STEPS
#endif

#ifndef TD_DEMO_POLICY_SEED
#define TD_DEMO_POLICY_SEED UINT32_C(2026072260)
#endif

static void allocate(TowerDefence *env) {
    td_init(env);
    env->observations = (float *)calloc(TD_OBS_SIZE, sizeof(float));
    env->actions = (float *)calloc(1, sizeof(float));
    env->rewards = (float *)calloc(1, sizeof(float));
    env->terminals = (float *)calloc(1, sizeof(float));
    env->action_mask = (unsigned char *)calloc(TD_NUM_ACTIONS, sizeof(unsigned char));
    if (env->observations == NULL || env->actions == NULL || env->rewards == NULL ||
        env->terminals == NULL || env->action_mask == NULL) {
        fprintf(stderr, "Failed to allocate tower_defence demo buffers\n");
        exit(1);
    }
}

static void free_allocated(TowerDefence *env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(env->action_mask);
    free(env->projectiles);
    env->projectiles = NULL;
    env->projectile_count = 0;
    env->projectile_capacity = 0;
}

static int td_demo_align8(int value) {
    return (value + 7) & ~7;
}

static int td_demo_expected_weight_count(int obs_size, int hidden_size, int num_layers,
                                         int action_count) {
    int count = 0;
    count = td_demo_align8(count + hidden_size * obs_size);
    count = td_demo_align8(count + (action_count + 1) * hidden_size);
    for (int layer = 0; layer < num_layers; layer++) {
        count = td_demo_align8(count + 3 * hidden_size * hidden_size);
    }
    return count;
}

static int td_demo_weights_compatible(const Weights *weights) {
    int expected = td_demo_expected_weight_count(TD_OBS_SIZE, 128, 2, TD_NUM_ACTIONS);
    return weights != NULL && weights->size - 7 == expected;
}

static int demo_action(TowerDefence *env, TdClient *client) {
    if (!td_human_control_active()) {
        return TD_ACTION_NOOP;
    }
    static const int tower_keys[TD_NUM_TOWER_TYPES] = {KEY_ONE, KEY_TWO, KEY_THREE};
    for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
        if (IsKeyPressed(tower_keys[kind])) {
            client->tower_kind = kind;
        }
    }
    if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) {
        return env->action_mask[TD_ACTION_TRIGGER_NEXT_ROUND] ? TD_ACTION_TRIGGER_NEXT_ROUND
                                                              : TD_ACTION_NOOP;
    }
    client->hover_slot = td_mouse_hover_slot();
    if (client->hover_slot < 0) {
        return TD_ACTION_NOOP;
    }
    int slot = client->hover_slot;
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
        int action = TD_ACTION_PLACE_FIRST + slot * TD_NUM_TOWER_TYPES + client->tower_kind;
        return env->action_mask[action] ? action : TD_ACTION_NOOP;
    }
    static const int upgrade_keys[TD_NUM_UPGRADE_PATHS] = {KEY_Q, KEY_W, KEY_E};
    for (int path = 0; path < TD_NUM_UPGRADE_PATHS; path++) {
        if (IsKeyPressed(upgrade_keys[path])) {
            int action = TD_ACTION_UPGRADE_SLOT_01_TOP + slot * TD_NUM_UPGRADE_PATHS + path;
            return env->action_mask[action] ? action : TD_ACTION_NOOP;
        }
    }
    if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON) || IsKeyPressed(KEY_X)) {
        int action = TD_ACTION_SELL_SLOT_01 + slot;
        return env->action_mask[action] ? action : TD_ACTION_NOOP;
    }
    return TD_ACTION_NOOP;
}

static void td_reset_policy_state(PufferNet *net) {
    if (net == NULL || net->mingru == NULL) {
        return;
    }
    size_t state_count = (size_t)net->mingru->num_layers * (size_t)net->mingru->batch_size *
                         (size_t)net->mingru->hidden_size;
    memset(net->mingru->state, 0, state_count * sizeof(*net->mingru->state));
}

static uint32_t td_demo_random(uint32_t *state) {
    uint32_t value = *state;
    if (value == 0) {
        value = UINT32_C(0x6d2b79f5);
    }
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static int td_demo_sample_masked_action(const float *logits, const unsigned char *mask,
                                        int action_count, uint32_t *rng) {
    int fallback = -1;
    float max_logit = -INFINITY;
    for (int action = 0; action < action_count; action++) {
        if (!mask[action]) {
            continue;
        }
        if (fallback < 0) {
            fallback = action;
        }
        if (isfinite(logits[action]) && logits[action] > max_logit) {
            max_logit = logits[action];
        }
    }
    if (!isfinite(max_logit)) {
        return fallback < 0 ? TD_ACTION_NOOP : fallback;
    }

    float total = 0.0f;
    for (int action = 0; action < action_count; action++) {
        if (mask[action] && isfinite(logits[action])) {
            total += expf(logits[action] - max_logit);
        }
    }
    if (!(total > 0.0f) || !isfinite(total)) {
        return fallback;
    }

    float uniform = (float)(td_demo_random(rng) >> 8) * (1.0f / 16777216.0f);
    float threshold = uniform * total;
    float cumulative = 0.0f;
    for (int action = 0; action < action_count; action++) {
        if (!mask[action] || !isfinite(logits[action])) {
            continue;
        }
        cumulative += expf(logits[action] - max_logit);
        if (threshold < cumulative) {
            return action;
        }
    }
    return fallback;
}

static int policy_action(TowerDefence *env, PufferNet *net, uint32_t *rng) {
    linear(net->encoder, env->observations);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    return td_demo_sample_masked_action(net->decoder->output, env->action_mask, TD_NUM_ACTIONS,
                                        rng);
}

int main(void) {
    TowerDefence env = {0};
    allocate(&env);
    env.max_episode_steps = TD_DEMO_MAX_EPISODE_STEPS;
    c_reset(&env);
    c_render(&env);

    Weights *weights =
        access(TD_DEMO_WEIGHT_PATH, R_OK) == 0 ? load_weights(TD_DEMO_WEIGHT_PATH) : NULL;
    if (weights && !td_demo_weights_compatible(weights)) {
        printf("Ignoring incompatible policy weights at %s: got %d floats for typed-placement "
               "obs/action contract\n",
               TD_DEMO_WEIGHT_PATH, weights->size - 7);
        free(weights);
        weights = NULL;
    } else if (!weights) {
        printf("No compatible policy weights at %s; hold SHIFT for manual control\n",
               TD_DEMO_WEIGHT_PATH);
    }
    int logit_sizes[1] = {TD_NUM_ACTIONS};
    PufferNet *net =
        weights ? make_puffernet(weights, 1, TD_OBS_SIZE, 128, 2, logit_sizes, 1) : NULL;
    env.client->manual_controls_enabled = 1;
    env.client->policy_loaded = net != NULL;

    int queued_manual_action = TD_ACTION_NOOP;
    int was_human_control = td_human_control_active();
    uint32_t policy_rng = TD_DEMO_POLICY_SEED;
    double accumulator = 0.0;
    while (!WindowShouldClose()) {
        int human_control = td_human_control_active();
        if (human_control) {
            int input_action = demo_action(&env, env.client);
            if (input_action != TD_ACTION_NOOP) {
                queued_manual_action = input_action;
            }
        } else if (was_human_control) {
            queued_manual_action = TD_ACTION_NOOP;
            td_reset_policy_state(net);
        }
        was_human_control = human_control;

        accumulator += fmin((double)GetFrameTime(), 4.0 * TD_DT);
        while (accumulator >= TD_DT) {
            if (net != NULL && !human_control) {
                env.actions[0] = (float)policy_action(&env, net, &policy_rng);
            } else {
                env.actions[0] = (float)queued_manual_action;
                queued_manual_action = TD_ACTION_NOOP;
            }
            c_step(&env);
            if (env.terminals[0] != 0.0f) {
                td_reset_policy_state(net);
            }
            accumulator -= TD_DT;
        }
        c_render(&env);
    }

    if (net) {
        free_puffernet(net);
    }
    free(weights);
    c_close(&env);
    free_allocated(&env);
    return 0;
}
