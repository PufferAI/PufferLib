#include <time.h>
#include "snake.h"
#include "puffernet.h"

int demo() {
    CSnake env = {
        .num_snakes = 256,
        .width = 640,
        .height = 360,
        .max_snake_length = 200,
        .food = 4096,
        .vision = 5,
        .leave_corpse_on_death = true,
        .reward_food = 1.0f,
        .reward_corpse = 0.5f,
        .reward_death = -1.0f,
    };
    allocate_csnake(&env);
    c_reset(&env);

    Weights* weights = load_weights("resources/snake/snake_weights.bin", 256773);
    int logit_sizes[] = {4};
    LinearLSTM* net = make_linearlstm(weights, env.num_snakes, 968, logit_sizes, 1);
    env.client = make_client(2, env.width, env.height);

    while (!WindowShouldClose()) {
        // User can take control of the first snake
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if (IsKeyDown(KEY_UP)    || IsKeyDown(KEY_W)) env.actions[0] = 0;
            if (IsKeyDown(KEY_DOWN)  || IsKeyDown(KEY_S)) env.actions[0] = 1;
            if (IsKeyDown(KEY_LEFT)  || IsKeyDown(KEY_A)) env.actions[0] = 2;
            if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) env.actions[0] = 3;
        } else {
            memset(net->obs, 0, env.num_snakes*968*sizeof(float));
            for (int i = 0; i < env.num_snakes*121; i++) {
                int obs = env.observations[i];
                net->obs[i*8 + obs] = 1.0f;
            }
            forward_linearlstm(net, net->obs, env.actions);
        }
        c_step(&env);
        c_render(&env);
    }
    free_linearlstm(net);
    free(weights);
    close_client(env.client);
    free_csnake(&env);
    return 0;
}

void test_performance(float test_time) {
    CSnake env = {
        .num_snakes = 1024,
        .width = 1280,
        .height = 720,
        .max_snake_length = 200,
        .food = 16384,
        .vision = 5,
        .leave_corpse_on_death = true,
        .reward_food = 1.0f,
        .reward_corpse = 0.5f,
        .reward_death = -1.0f,
    };
    allocate_csnake(&env);
    c_reset(&env);

    int start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        for (int j = 0; j < env.num_snakes; j++) {
            env.actions[j] = rand()%4;
        }
        c_step(&env);
        i++;
    }
    int end = time(NULL);
    free_csnake(&env);
    printf("SPS: %f\n", (float)env.num_snakes*i / (end - start));
}

int main() {
    demo();
    // test_performance(30);
    return 0;
}
