#include "tower_defence.h"
#include "puffernet.h"
#include <unistd.h>

#ifndef TD_DEMO_WEIGHT_PATH
#define TD_DEMO_WEIGHT_PATH "resources/tower_defence/tower_defence_weights.bin"
#endif

static int demo_action(TowerDefence* env) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        int slot = env->hover_slot >= 0 ? env->hover_slot : 0;
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) return 1 + slot;
        if (IsKeyPressed(KEY_Q)) return TD_ACTION_UPGRADE_SLOT_01_TOP + slot * TD_NUM_UPGRADE_PATHS;
        if (IsKeyPressed(KEY_W)) return TD_ACTION_UPGRADE_SLOT_01_TOP + slot * TD_NUM_UPGRADE_PATHS + 1;
        if (IsKeyPressed(KEY_E)) return TD_ACTION_UPGRADE_SLOT_01_TOP + slot * TD_NUM_UPGRADE_PATHS + 2;
        if (IsMouseButtonPressed(MOUSE_RIGHT_BUTTON) || IsKeyPressed(KEY_X)) {
            return TD_ACTION_SELL_SLOT_01 + slot;
        }
        if (IsKeyPressed(KEY_SPACE) || IsKeyPressed(KEY_ENTER)) return TD_ACTION_TRIGGER_NEXT_ROUND;
        return TD_ACTION_NOOP;
    }
    for (int action = 1; action < TD_ACTION_SELL_SLOT_01; action++) {
        if (env->valid_actions[action]) return action;
    }
    if (env->valid_actions[TD_ACTION_TRIGGER_NEXT_ROUND]) return TD_ACTION_TRIGGER_NEXT_ROUND;
    for (int action = TD_ACTION_SELL_SLOT_01; action < TD_ACTION_TRIGGER_NEXT_ROUND; action++) {
        if (env->valid_actions[action]) return action;
    }
    return TD_ACTION_NOOP;
}

static void set_action(TowerDefence* env, int action) {
#if TD_USE_FACTORED_ACTIONS
    env->actions[0] = TD_FACTOR_VERB_NOOP;
    env->actions[1] = 0;
    env->actions[2] = 0;
    if (action >= 1 && action <= TD_NUM_PLACEMENT_SLOTS) {
        env->actions[0] = TD_FACTOR_VERB_PLACE;
        env->actions[1] = action - 1;
    } else if (action >= TD_ACTION_UPGRADE_SLOT_01_TOP && action < TD_ACTION_SELL_SLOT_01) {
        int idx = action - TD_ACTION_UPGRADE_SLOT_01_TOP;
        env->actions[0] = TD_FACTOR_VERB_UPGRADE;
        env->actions[1] = idx / TD_NUM_UPGRADE_PATHS;
        env->actions[2] = idx % TD_NUM_UPGRADE_PATHS;
    } else if (action >= TD_ACTION_SELL_SLOT_01 && action < TD_ACTION_TRIGGER_NEXT_ROUND) {
        env->actions[0] = TD_FACTOR_VERB_SELL;
        env->actions[1] = action - TD_ACTION_SELL_SLOT_01;
    } else if (action == TD_ACTION_TRIGGER_NEXT_ROUND) {
        env->actions[0] = TD_FACTOR_VERB_TRIGGER;
    }
#else
    env->actions[0] = action;
#endif
}

int main() {
    TowerDefence env = {0};
    allocate(&env);
    c_reset(&env);
    c_render(&env);

    Weights* weights = access(TD_DEMO_WEIGHT_PATH, R_OK) == 0
        ? load_weights(TD_DEMO_WEIGHT_PATH)
        : NULL;
#if TD_USE_FACTORED_ACTIONS
    int logit_sizes[3] = {TD_FACTOR_VERB_COUNT, TD_NUM_PLACEMENT_SLOTS, TD_NUM_UPGRADE_PATHS};
    int num_atns = 3;
#else
    int logit_sizes[1] = {TD_NUM_ACTIONS};
    int num_atns = 1;
#endif
    PufferNet* net = weights ? make_puffernet(weights, 1, TD_OBS_SIZE, 128, 2, logit_sizes, num_atns) : NULL;

    while (!WindowShouldClose()) {
        if (net && !IsKeyDown(KEY_LEFT_SHIFT)) {
            forward_puffernet(net, env.observations, env.actions);
        } else {
            set_action(&env, demo_action(&env));
        }
        c_step(&env);
        if (IsKeyDown(KEY_LEFT_SHIFT)) set_action(&env, TD_ACTION_NOOP);
        c_render(&env);
    }

    if (net) free_puffernet(net);
    free(weights);
    c_close(&env);
    free_allocated(&env);
    return 0;
}
