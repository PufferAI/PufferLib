#include "robocode.h"
#include <time.h>

void performance_test() {
    long test_time = 10;
    Robocode env = {
        .num_agents = 2,
        .num_bots = 0,
        .width = 800,
        .height = 600,
        .reward_damage = 0.01f,
        .reward_spot = 0.001f,
        .bot_policy = 3,  // BOT_WAVE_SURFER
        .max_ticks = 3000,
        .rng = 42,
    };
    allocate_env(&env);
    c_reset(&env);

    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand_r(&env.rng) % 4;
        env.actions[1] = rand_r(&env.rng) % 9;
        env.actions[2] = rand_r(&env.rng) % 11;
        env.actions[3] = rand_r(&env.rng) % 11;
        env.actions[4] = (rand_r(&env.rng) % 6) > 4 ? 1.0f : 0.0f;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", (long)i*env.num_agents / (end - start));
    c_close(&env);
}

void demo(void) {
    Robocode env = {
        .num_agents = 1,
        .num_bots = 1,
        .reward_damage = 0.01,
        .width = 800,
        .height = 600,
        .max_ticks = 512,
    };
    allocate_env(&env);
    c_reset(&env);

    env.client = make_client(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        env.actions[0] = 2;
        env.actions[1] = 4;
        env.actions[2] = 5;
        env.actions[3] = 5;
        env.actions[4] = 0;

        if (IsKeyPressed(KEY_ESCAPE)) break;
        if (IsKeyDown(KEY_W)) env.actions[0] = 3.0f;
        if (IsKeyDown(KEY_S)) env.actions[0] = 1.0f;
        if (IsKeyDown(KEY_A)) env.actions[1] = 3.0f;
        if (IsKeyDown(KEY_D)) env.actions[1] = 5.0f;
        if (IsKeyDown(KEY_Q)) env.actions[2] = 4.0f;
        if (IsKeyDown(KEY_E)) env.actions[2] = 6.0f;
        if (IsKeyDown(KEY_LEFT)) env.actions[3] = 0.0f;
        if (IsKeyDown(KEY_RIGHT)) env.actions[3] = 8.0f;
        if (IsKeyDown(KEY_SPACE)) env.actions[4] = 1.0f;

        c_step(&env);
        c_render(&env);
    }
    c_close(&env);
    CloseWindow();
}

int main() {
    demo();
    //performance_test();
    return 0;
}
