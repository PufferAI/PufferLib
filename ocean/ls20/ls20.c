#include <inttypes.h>
#include <stdio.h>

#include "ls20.h"

static const char* WINNING_LEVELS[LS20_LEVEL_COUNT] = {
    "LLLUUUURRRUUU",
    "URUUUUURRDRDDDDDDDLLRURLRUUUUUUULLLLLLDLDDDDD",
    "UUUUUUUULDDDDDDDDUUULLUUURRRDDRRRRUUULRLDRDDDDDDD",
    "LLLDDDLLLRRUUULLUDDUDUDUDUDUDUDUULLUDDULLUUUDLULUUUURURUULLL",
    "ULUULDLULRLRLRUDLRLDDLLLULRLRDLDDDDDRDURRULRDDDRRRRRRDUU",
    ("UUDULRRRRUUULLRRDDDDLLLURRRUUUUUUUDDRRUURDDDUUULUDLLLRLRLLLL"
     "DDDRRRDULDLLUUUUUUURLRRRRLLLULDDDDDDDRRDURRRRUUUUURRUURDDDDD"),
    "LLDDDDDRLRLRRLDURLRLRLRLDUULUURRRRUDRRUURRUUUUUURDLDDDDDLLUDDDD",
};

static int action_from_char(char action) {
    if (action == 'U') return ACTION1;
    if (action == 'D') return ACTION2;
    if (action == 'L') return ACTION3;
    assert(action == 'R');
    return ACTION4;
}

static uint64_t hash_observation(uint64_t hash, const unsigned char* observation) {
    for (int i = 0; i < LS20_OBS_SIZE; i++) {
        hash = (hash ^ observation[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static int smoke(void) {
    unsigned char observations[LS20_OBS_SIZE] = {0};
    float action = 0.0f;
    float reward = 0.0f;
    float terminal = 0.0f;
    unsigned char action_mask[LS20_ACTION_COUNT] = {0};
    Ls20 env = {
        .num_agents = 1,
        .reset_enabled = true,
        .observations = observations,
        .action_mask = action_mask,
        .actions = &action,
        .rewards = &reward,
        .terminals = &terminal,
    };
    c_reset(&env);

    int rewards = 0;
    // FNV-1a of Python's initial and settled frames; C autoresets after the final frame.
    uint64_t observation_hash = hash_observation(
        UINT64_C(14695981039346656037), observations
    );
    for (int level = 0; level < LS20_LEVEL_COUNT; level++) {
        for (const char* step = WINNING_LEVELS[level]; *step; step++) {
            action = (float)action_from_char(*step);
            c_step(&env);
            rewards += (int)reward;
            bool level_ended = step[1] == '\0';
            bool game_ended = level_ended && level == LS20_LEVEL_COUNT - 1;
            if (!game_ended) {
                observation_hash = hash_observation(observation_hash, observations);
            }
            if ((reward == 1.0f) != level_ended || (terminal == 1.0f) != game_ended) {
                fprintf(stderr, "smoke failed at level %d\n", level + 1);
                return 1;
            }
        }
    }

    if (rewards != LS20_LEVEL_COUNT || terminal != 1.0f
            || env.log.n != 1.0f || env.log.score != 7.0f) {
        fprintf(stderr, "smoke failed: rewards=%d terminal=%.0f score=%.0f\n",
            rewards, terminal, env.log.score);
        return 1;
    }
    if (observation_hash != UINT64_C(0x8afb357ef4212d60)) {
        fprintf(stderr, "smoke failed: observation hash=%" PRIx64 "\n", observation_hash);
        return 1;
    }

    c_reset(&env);
    for (const char* step = WINNING_LEVELS[0]; *step; step++) {
        action = (float)action_from_char(*step);
        c_step(&env);
    }
    action = ACTION1;
    c_step(&env);
    action = RESET;
    c_step(&env);
    if (env.level_index != 1 || env.levels_completed != 1 || env.level_action_count != 0
            || env.episode_length != 15) {
        fputs("smoke failed: level reset\n", stderr);
        return 1;
    }
    c_step(&env);
    if (env.level_index != 0 || env.levels_completed != 0 || env.episode_length != 16) {
        fputs("smoke failed: game reset\n", stderr);
        return 1;
    }

    env.reset_enabled = false;
    c_reset(&env);
    action = ACTION2;
    c_step(&env);
    action = RESET;
    c_step(&env);
    if (env.energy != LS20_MAX_ENERGY - 1 || env.level_action_count != 1
            || action_mask[RESET] != 0) {
        fputs("smoke failed: disabled reset\n", stderr);
        return 1;
    }

    env.reset_enabled = true;
    c_reset(&env);
    action = ACTION2;
    for (int i = 0; i < 42; i++) c_step(&env);
    if (env.lives != 3 || env.energy != 0 || terminal != 0.0f) {
        fputs("smoke failed: energy boundary\n", stderr);
        return 1;
    }
    c_step(&env);
    if (env.lives != 2 || env.energy != LS20_MAX_ENERGY || terminal != 0.0f) {
        fputs("smoke failed: life reset\n", stderr);
        return 1;
    }
    for (int i = 0; i < 86; i++) c_step(&env);
    if (terminal != 1.0f || env.log.n != 2.0f || env.level_index != 0
            || env.lives != 3 || env.energy != LS20_MAX_ENERGY) {
        fputs("smoke failed: lives terminal\n", stderr);
        return 1;
    }

    unsigned int state = 1;
    for (int i = 0; i < 100000; i++) {
        state = state * 1664525u + 1013904223u;
        action = (float)(state % 5);
        c_step(&env);
    }
    puts("LS20 smoke passed");
    c_close(&env);
    return 0;
}

static void play(void) {
    unsigned char observations[LS20_OBS_SIZE] = {0};
    float action = 0.0f;
    float reward = 0.0f;
    float terminal = 0.0f;
    Ls20 env = {
        .num_agents = 1,
        .observations = observations,
        .actions = &action,
        .rewards = &reward,
        .terminals = &terminal,
    };
    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        int next_action = -1;
        if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) next_action = ACTION1;
        if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) next_action = ACTION2;
        if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) next_action = ACTION3;
        if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) next_action = ACTION4;
        if (IsKeyPressed(KEY_R)) next_action = RESET;
        if (next_action >= 0) {
            action = (float)next_action;
            c_step(&env);
        }
        c_render(&env);
    }
    c_close(&env);
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--smoke") == 0) return smoke();
    play();
    return 0;
}
