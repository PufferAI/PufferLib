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

static void demo(void) {
    unsigned char observations[LS20_OBS_SIZE] = {0};
    float action = 0.0f;
    float reward = 0.0f;
    float terminal = 0.0f;
    Ls20 env = {
        .num_agents = 1,
        .fps = LS20_DEFAULT_FPS,
        .observations = observations,
        .actions = &action,
        .rewards = &reward,
        .terminals = &terminal,
    };
    c_reset(&env);
    c_render(&env);
    for (int level = 0; level < LS20_LEVEL_COUNT; level++) {
        for (const char* step = WINNING_LEVELS[level]; *step; step++) {
            action = (float)action_from_char(*step);
            c_step(&env);
            c_render(&env);
        }
    }
    c_close(&env);
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
    if (argc == 2 && strcmp(argv[1], "--demo") == 0) {
        demo();
        return 0;
    }
    play();
    return 0;
}
