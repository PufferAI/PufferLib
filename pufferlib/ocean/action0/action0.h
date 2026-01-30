/*
 * Action0 Environment
 *
 * A simple credit assignment test. The agent must take action 0
 * on the first step to win. The episode terminates at step 128 and
 * the reward is given only at termination.
 *
 * Observation: Box(0, 1, (1,)) - always 1
 * Action: Discrete(2)
 * Win condition: action 0 on step 1
 * Horizon: 128 steps
 */

#include <stdlib.h>
#include <string.h>

// Log struct - only floats, ends with n
typedef struct {
    float score;
    float n;
} Log;

typedef struct {
    Log log;                     // Required field
    float* observations;         // Required field
    int* actions;                // Required field
    float* rewards;              // Required field
    unsigned char* terminals;    // Required field
    int horizon;
    int tick;
    int won;
} Action0;

void c_reset(Action0* env) {
    env->observations[0] = 1.0f;
    env->tick = 0;
    env->won = 0;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
}

void c_step(Action0* env) {
    env->tick++;

    // Check if first step and correct action
    if (env->tick == 1 && env->actions[0] == 0) {
        env->won = 1;
    }

    // Always reset reward/terminal first
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    // Check if episode is done
    if (env->tick >= env->horizon) {
        env->rewards[0] = env->won ? 1.0f : 0.0f;
        env->terminals[0] = 1;
        env->log.score += env->won ? 1.0f : 0.0f;
        env->log.n += 1.0f;
        // Reset state for next episode but DON'T overwrite reward/terminal
        env->tick = 0;
        env->won = 0;
        env->observations[0] = 1.0f;
    }
}

void c_render(Action0* env) {
    // No rendering for this simple env
}

void c_close(Action0* env) {
    // Nothing to clean up
}
