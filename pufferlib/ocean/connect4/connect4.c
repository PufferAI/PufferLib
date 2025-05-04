#include "connect4.h"
#include "puffernet.h"
#include "time.h"

const unsigned char NOOP = 8;

void interactive() {
    Weights* weights = load_weights("resources/connect4_weights.bin", 138632);
    LinearLSTM* net = make_linearlstm(weights, 1, 42, 7);

    CConnect4 env = {
    };
    allocate_cconnect4(&env);
    c_reset(&env);
 
    env.client = make_client();
    float observations[42] = {0};
    int actions[1] = {0};

    int tick = 0;
    while (!WindowShouldClose()) {
        env.actions[0] = NOOP;
        // user inputs 1 - 7 key pressed
        if (IsKeyDown(KEY_LEFT_SHIFT)) {
            if(IsKeyPressed(KEY_ONE)) env.actions[0] = 0;
            if(IsKeyPressed(KEY_TWO)) env.actions[0] = 1;
            if(IsKeyPressed(KEY_THREE)) env.actions[0] = 2;
            if(IsKeyPressed(KEY_FOUR)) env.actions[0] = 3;
            if(IsKeyPressed(KEY_FIVE)) env.actions[0] = 4;
            if(IsKeyPressed(KEY_SIX)) env.actions[0] = 5;
            if(IsKeyPressed(KEY_SEVEN)) env.actions[0] = 6;
        } else if (tick % 30 == 0) {
            for (int i = 0; i < 42; i++) {
                observations[i] = env.observations[i];
            }
            forward_linearlstm(net, (float*)&observations, (int*)&actions);
            env.actions[0] = actions[0];
        }

        tick = (tick + 1) % 60;
        if (env.actions[0] >= 0 && env.actions[0] <= 6) {
            c_step(&env);
        }

        c_render(&env);
    }
    free_linearlstm(net);
    free(weights);
    close_client(env.client);
    free_allocated_cconnect4(&env);
}

void performance_test() {
    long test_time = 10;
    CConnect4 env = {
    };
    allocate_cconnect4(&env);
    c_reset(&env);
 
    long start = time(NULL);
    int i = 0;
    while (time(NULL) - start < test_time) {
        env.actions[0] = rand() % 7;
        c_step(&env);
        i++;
    }
    long end = time(NULL);
    printf("SPS: %ld\n", i / (end - start));
    free_allocated_cconnect4(&env);
}

int main() {
    // performance_test();
    interactive();
    return 0;
}
